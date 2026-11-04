"""Public benchmark loaders with aligned utility, cost, and candidate columns."""

import json
import logging
import os
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

from utils.sources import (
    DATA_SOURCES,
    download_asset,
    file_sha256,
    load_verified_pickle,
    verify_file,
)


def _safe_archive_target(target_dir: str, member_name: str) -> Path:
    """Resolve an archive member without allowing it to escape target_dir."""
    normalized = member_name.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    if (
        not normalized
        or member_path.is_absolute()
        or ".." in member_path.parts
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise ValueError(f"Unsafe archive member path: {member_name!r}")

    root = Path(target_dir).resolve()
    destination = root.joinpath(*member_path.parts).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Archive member escapes target directory: {member_name!r}") from exc
    return destination


def safe_extract_tar(archive: tarfile.TarFile, target_dir: str) -> None:
    """Extract regular files/directories only, after validating every member."""
    members = archive.getmembers()
    destinations = []
    for member in members:
        destination = _safe_archive_target(target_dir, member.name)
        if not (member.isfile() or member.isdir()):
            raise ValueError(
                f"Unsupported TAR member type for {member.name!r}; "
                "links and special files are not allowed."
            )
        destinations.append((member, destination))

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    for member, destination in destinations:
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Cannot read TAR member: {member.name!r}")
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)


def safe_extract_zip(archive: zipfile.ZipFile, target_dir: str) -> None:
    """Extract ZIP members only after validating paths and rejecting symlinks."""
    members = archive.infolist()
    destinations = []
    for member in members:
        destination = _safe_archive_target(target_dir, member.filename)
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise ValueError(f"ZIP symlink members are not allowed: {member.filename!r}")
        destinations.append((member, destination))

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    for member, destination in destinations:
        if member.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)


class BaseDatasetLoader:
    """Base class for dataset loaders.

    Set `hf_id` and `default_target_path` in subclasses. The default Hugging Face
    downloader also needs a full commit hash in `source_metadata["revision"]`
    or the built-in source registry. Alternatively, override `download()`.
    """

    hf_id: str = ""
    default_target_path: str = "./data"
    name: str = "base"
    source_metadata: dict[str, Any] | None = None
    args: dict

    def __init__(self, target_path: str | None, args: dict):
        self.target_path = target_path or self.default_target_path
        self.args = args
        self.source = deepcopy(
            self.source_metadata
            if self.source_metadata is not None
            else DATA_SOURCES.get(self.hf_id, {})
        )
        if self.source and "dataset" in args:
            args["dataset"]["source"] = {"repo_id": self.hf_id, **deepcopy(self.source)}

    def download(self) -> Any:
        """Load a Hugging Face dataset at the declared commit revision."""
        revision = self.source.get("revision")
        if (
            not self.hf_id
            or not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", revision)
        ):
            raise ValueError(
                "The default dataset downloader requires hf_id and a full 40-character "
                'commit hash in source_metadata["revision"] or DATA_SOURCES; '
                "set these on the loader subclass or override download()."
            )
        logging.info("[utils.data.py] Loading dataset %s from HF id %s", self.name, self.hf_id)
        dataset = load_dataset(self.hf_id, revision=revision, cache_dir=self.target_path)
        return dataset

    def normalize_cost_dataframe(
        self, df: pd.DataFrame, min_bound: float = 0.1, epsilon: float = 0.1
    ) -> pd.DataFrame:
        cost_columns = [col for col in df.columns if "cost" in col]

        cost_values = df[cost_columns].values
        log_values = np.log1p(cost_values)

        row_mins = log_values.min(axis=1)[:, np.newaxis]
        row_maxs = log_values.max(axis=1)[:, np.newaxis]
        row_ranges = row_maxs - row_mins

        soft_min = row_mins - epsilon * row_ranges
        soft_max = row_maxs + epsilon * row_ranges
        soft_ranges = soft_max - soft_min

        fractions = np.divide(
            log_values - soft_min,
            soft_ranges,
            out=np.full_like(log_values, 0.5, dtype=np.float64),
            where=soft_ranges > 0,
        )
        normalized_values = min_bound + (1 - 2 * min_bound) * fractions

        df[cost_columns] = normalized_values
        return df

    def process(self, dataset: Any) -> Any:
        """
        Optional processing step.
        By default returns the dataset unchanged.
        Subclasses may return either:
          - a DataFrame (or dataset-like), or
          - a tuple (model_list, dataframe)
        """
        return dataset

    def split(self, dataset: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple train/test split according to args["dataset"]["split"]["ratios"].
        Returns: (train_df, test_df)
        """
        self._split_ratios()
        split_mode = self.args["dataset"]["split"]["mode"]
        if split_mode == "in-domain":
            return self.split_in(dataset)
        if split_mode == "out-of-domain":
            return self.split_out(dataset)
        raise ValueError(f"Unknown split mode: {split_mode}")

    def _split_ratios(self) -> tuple[float, float]:
        ratios = self.args["dataset"]["split"]["ratios"]
        if set(ratios) != {"train", "test"}:
            raise ValueError("Split ratios must explicitly contain only train and test")
        train, test = float(ratios["train"]), float(ratios["test"])
        if not (np.isfinite(train) and np.isfinite(test) and 0 < train < 1 and 0 < test < 1):
            raise ValueError("Train/test ratios must be finite and strictly between 0 and 1")
        if not np.isclose(train + test, 1.0, rtol=0, atol=1e-9):
            raise ValueError("Train/test ratios must sum to one")
        return train, test

    def split_in(self, dataset: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        """In-domain split: convert to pandas and return train/test DataFrames."""
        if hasattr(dataset, "to_pandas"):
            df = dataset.to_pandas()
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            df = pd.DataFrame(dataset)

        train_ratio, _ = self._split_ratios()
        seed = self.args["seed"]

        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_train = int(len(df) * train_ratio)
        train_df = df.iloc[:n_train].reset_index(drop=True)
        test_df = df.iloc[n_train:].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            raise ValueError("The split must contain non-empty training and test partitions")
        return train_df, test_df

    def split_out(self, dataset: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Partition evaluation tasks using the benchmark's grouped split policy."""
        df = dataset.to_pandas() if hasattr(dataset, "to_pandas") else pd.DataFrame(dataset)
        if "eval_name" not in df:
            raise ValueError("Out-of-domain splitting requires an eval_name column")
        train_ratio, test_ratio = self._split_ratios()
        counts = df.groupby("eval_name").size()
        order = np.random.RandomState(self.args["seed"]).permutation(len(counts))
        train_names = []
        train_count = 0
        target_count = len(df) * train_ratio
        test_threshold = len(df) * test_ratio * 0.8
        for index in order:
            name, count = counts.index[index], int(counts.iloc[index])
            if count > test_threshold or train_count + count <= target_count * 1.1:
                train_names.append(name)
                train_count += count
        mask = df["eval_name"].isin(train_names)
        train, test = df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(
                "Task-level split produced an empty partition; use more tasks or an in-domain split"
            )
        return train, test

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, Any | None]:
        """
        Run full pipeline: download -> process -> split.
        process() may return either:
          - dataframe-like, or
          - (model_list, dataframe)
        Returns: (train_df, test_df, model_list_or_None)
        """
        dataset = self.download()
        processed = self.process(dataset)

        if isinstance(processed, tuple) and len(processed) == 2:
            model_list, df = processed
        else:
            model_list = None
            df = processed

        train_df, test_df = self.split(df)
        return train_df, test_df, model_list


class RouterbenchLoader(BaseDatasetLoader):
    name = "Routerbench"
    hf_id = "withmartian/routerbench"
    default_target_path = "./data/routerbench"

    def download(self) -> str:
        """Download the pinned and SHA-256-verified official RouterBench pickle."""
        return str(download_asset(self.hf_id, "routerbench_raw.pkl", self.target_path, self.source))

    def process(self, dataset: Any) -> Any:
        """
        Load the pickle, normalize columns, pivot to wide format.
        Returns: (model_list, result_dataframe)
        """

        model_list = [
            "WizardLM/WizardLM-13B-V1.2",
            "claude-instant-v1",
            "claude-v1",
            "claude-v2",
            "gpt-3.5-turbo-1106",
            "gpt-4-1106-preview",
            "meta/code-llama-instruct-34b-chat",
            "meta/llama-2-70b-chat",
            "mistralai/mistral-7b-chat",
            "mistralai/mixtral-8x7b-chat",
            "zero-one-ai/Yi-34B-Chat",
        ]
        logging.info("[utils.data.py] Model List: %s", model_list)

        source_path = Path(dataset)
        expected = self.source["sha256"]["routerbench_raw.pkl"]
        parquet_path = Path(self.target_path) / f"routerbench_raw_{expected[:16]}.parquet"
        manifest_path = parquet_path.with_suffix(".manifest.json")
        if parquet_path.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest["source_sha256"] != expected:
                raise ValueError("RouterBench Parquet source hash mismatch")
            verify_file(parquet_path, manifest["parquet_sha256"])
            df = pd.read_parquet(parquet_path)
        else:
            raw_data = load_verified_pickle(source_path, expected)
            df = pd.DataFrame(raw_data)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = parquet_path.with_suffix(f".{os.getpid()}.tmp.parquet")
            df.to_parquet(temporary, index=False)
            parquet_hash = file_sha256(temporary)
            os.replace(temporary, parquet_path)
            temporary_manifest = manifest_path.with_suffix(f".{os.getpid()}.tmp")
            temporary_manifest.write_text(
                json.dumps({"source_sha256": expected, "parquet_sha256": parquet_hash})
            )
            os.replace(temporary_manifest, manifest_path)

        original_col_by_lower = {col.lower(): col for col in df.columns}

        def resolve_column(name_candidates):
            for cand in name_candidates:
                key = cand.lower()
                if key in original_col_by_lower:
                    return original_col_by_lower[key]
            return None

        sample_id_col = resolve_column(["sample_id", "sampleid", "id", "sample"])
        model_name_col = resolve_column(["model_name", "model", "model_name_str", "modelId"])
        prompt_col = resolve_column(
            ["prompt", "input", "instruction", "query", "context", "prompt_text"]
        )
        eval_name_col = resolve_column(["eval_name", "evaluation", "eval", "task", "eval_name_str"])
        performance_col = resolve_column(["performance", "score", "eval_score"])
        cost_col = resolve_column(["cost", "latency", "time", "inference_cost", "cost_usd"])
        response_col = resolve_column(
            ["model_response", "response", "output", "model_output", "answer"]
        )

        required = {
            "sample_id": sample_id_col,
            "model": model_name_col,
            "utility": performance_col,
            "cost": cost_col,
            "prompt": prompt_col,
        }
        missing = [name for name, column in required.items() if column is None]
        if missing:
            raise ValueError(f"RouterBench is missing required columns: {missing}")
        df[model_name_col] = df[model_name_col].astype(str)
        df = df[df[model_name_col].isin(model_list)].copy()

        selected_cols = [sample_id_col, model_name_col, performance_col, cost_col]
        if prompt_col:
            selected_cols.append(prompt_col)
        if eval_name_col:
            selected_cols.append(eval_name_col)
        if response_col:
            selected_cols.append(response_col)

        df = df[selected_cols].copy()

        perf_table = df.pivot_table(
            index=sample_id_col, columns=model_name_col, values=performance_col, aggfunc="first"
        )
        cost_table = df.pivot_table(
            index=sample_id_col, columns=model_name_col, values=cost_col, aggfunc="first"
        )
        resp_table = (
            df.pivot_table(
                index=sample_id_col, columns=model_name_col, values=response_col, aggfunc="first"
            )
            if response_col
            else pd.DataFrame(index=perf_table.index)
        )

        perf_table = perf_table.reindex(columns=model_list)
        cost_table = cost_table.reindex(columns=model_list)
        resp_table = resp_table.reindex(columns=model_list)

        perf_table.columns = [f"model_{i}_performance" for i in range(len(perf_table.columns))]
        cost_table.columns = [f"model_{i}_cost" for i in range(len(cost_table.columns))]
        resp_table.columns = [f"model_{i}_response" for i in range(len(resp_table.columns))]

        extra_series = []
        if prompt_col:
            prompt_series = df.groupby(sample_id_col)[prompt_col].first().rename("prompt")
            extra_series.append(prompt_series)
        if eval_name_col:
            eval_series = df.groupby(sample_id_col)[eval_name_col].first().rename("eval_name")
            extra_series.append(eval_series)

        result_parts = extra_series + [perf_table, cost_table, resp_table]
        result_df = pd.concat(result_parts, axis=1).reset_index()
        return model_list, result_df


class MixinstructLoader(BaseDatasetLoader):
    name = "Mixinstruct"
    hf_id = "llm-blender/mix-instruct"
    default_target_path = "./data/mixinstruct"

    def process(self, dataset: Any) -> Any:
        logging.info("[utils.data.py][Mixinstruct] Starting processing...")

        model_list = [
            "alpaca-native",
            "chatglm-6b",
            "dolly-v2-12b",
            "flan-t5-xxl",
            "koala-7B-HF",
            "llama-7b-hf-baize-lora-bf16",
            "moss-moon-003-sft",
            "mpt-7b",
            "mpt-7b-instruct",
            "oasst-sft-4-pythia-12b-epoch-3.5",
            "stablelm-tuned-alpha-7b",
            "vicuna-13b-1.1",
        ]
        logging.info(
            "[utils.data.py][Mixinstruct] Model list (%d): %s", len(model_list), model_list
        )
        model_costs = {
            0: 13,
            1: 6,
            2: 12,
            3: 11,
            4: 7,
            5: 7,
            6: 16,
            7: 7,
            8: 7,
            9: 12,
            10: 7,
            11: 13,
        }
        max_cost = max(model_costs.values())
        model_costs_norm = {i: cost / max_cost for i, cost in model_costs.items()}
        rows = []
        logging.info(
            "[utils.data.py][Mixinstruct] Extracting performance, cost and response data..."
        )
        for split_name in dataset:
            split_ds = dataset[split_name]
            for item in split_ds:
                eval_name = "unknown"
                raw_id = item.get("id", None)
                if raw_id is not None:
                    id_str = str(raw_id)
                    match = re.match(r"^([^\d]+)", id_str)
                    if match:
                        eval_name = match.group(1).rstrip("/")
                row = {
                    "sample_id": item["id"],
                    "prompt": item["instruction"] + " " + item.get("input", ""),
                    "eval_name": eval_name,
                }
                for i in range(len(model_list)):
                    row[f"model_{i}_performance"] = None
                    row[f"model_{i}_response"] = None
                    row[f"model_{i}_cost"] = None

                for cand in item.get("candidates", []):
                    model_name = cand.get("model")
                    if model_name in model_list:
                        model_idx = model_list.index(model_name)
                        score = cand.get("scores", {}).get("bertscore")
                        response = cand.get("text")
                        row[f"model_{model_idx}_performance"] = score
                        row[f"model_{model_idx}_response"] = response
                for i, cost in model_costs_norm.items():
                    row[f"model_{i}_cost"] = cost

                rows.append(row)
        result_df = pd.DataFrame(rows)

        logging.info(
            "[utils.data.py][Mixinstruct] Final dataframe shape: %s | columns: %d",
            result_df.shape,
            len(result_df.columns),
        )
        return model_list, result_df


class MMRBenchLoader(BaseDatasetLoader):
    name = "MMRBench"
    hf_id = "gh0stHunter/MMR-Bench"
    default_target_path = "./data/MMRBench"

    def download(self) -> str:
        """Download fixed official data and verify the archive before extraction."""
        download_asset(self.hf_id, "MMR-Bench.csv", self.target_path, self.source)
        tar_path = download_asset(self.hf_id, "images.tar.gz", self.target_path, self.source)
        marker = Path(self.target_path) / ".images-extracted"
        archive_hash = self.source["sha256"]["images.tar.gz"]
        if not marker.is_file() or marker.read_text() != archive_hash:
            with tarfile.open(tar_path, "r:gz") as archive:
                safe_extract_tar(archive, self.target_path)
            marker.write_text(archive_hash)
        return self.target_path

    def process(self, dataset: Any) -> Any:
        model_dict = {
            0: "Qwen2.5-VL-3B-Instruct",
            1: "Qwen2.5-VL-72B-Instruct",
            2: "InternVL3-78B",
            3: "Gemma3-4B",
            4: "Qwen2.5-VL-7B-Instruct",
            5: "Claude3-7V_Sonnet",
            6: "gpt-5-nano-2025-08-07",
            7: "GeminiPro2-5",
            8: "gpt-5-2025-08-07",
            9: "GeminiFlash2-5",
        }

        model_list = [model_dict[i] for i in range(len(model_dict))]
        logging.info(f"[utils.data.py] Model List: {model_list}")

        data_dir = self.target_path

        csv_files = os.path.join(data_dir, "MMR-Bench.csv")

        df = pd.read_csv(csv_files)

        def get_image_path(dataset_idx):
            match = re.search(r"^(.*)_([^_]+)$", dataset_idx)
            if match is None:
                raise ValueError(f"Invalid MMR-Bench sample identifier: {dataset_idx!r}")
            subdir = match.group(1)
            qid = match.group(2)
            if subdir == "SEEDBench2_Plus":
                subdir = "SEEDBenchv2Plus"

            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = f"{subdir}/{qid}{ext}"
                full_path = f"{data_dir}/{img_path}"
                if os.path.exists(full_path):
                    return subdir, full_path.replace("\\", "/")
            raise FileNotFoundError(f"No image found for MMR-Bench sample {dataset_idx!r}")

        eval_name, image_path = zip(*df["dataset_idx"].apply(get_image_path))
        df["image_path"] = image_path
        df["eval_name"] = eval_name

        logging.info(f"[utils.data.py] Combined DataFrame shape: {df.shape}")

        new_columns = ["sample_id", "prompt", "answer", "eval_name", "image_path"]
        for model_id, model_name in model_dict.items():
            new_columns.append(f"model_{model_id}_performance")
            new_columns.append(f"model_{model_id}_cost")
            new_columns.append(f"model_{model_id}_prediction")

        new_df = pd.DataFrame(columns=new_columns)
        column_mapping = {
            "sample_id": "dataset_idx",
            "prompt": "question",
            "answer": "answer",
            "eval_name": "eval_name",
            "image_path": "image_path",
        }
        for new_col, old_col in column_mapping.items():
            new_df[new_col] = df[old_col]
        for model_id, model_name in model_dict.items():
            perf_col = f"{model_name}_correct"
            cost_col = f"{model_name}_cost"
            pred_col = f"{model_name}_prediction"
            perf_values = df[perf_col]
            cost_values = df[cost_col]
            pred_values = df[pred_col]
            performance = (
                perf_values.astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})
            )
            new_df[f"model_{model_id}_performance"] = performance
            new_df[f"model_{model_id}_cost"] = cost_values
            new_df[f"model_{model_id}_prediction"] = pred_values
        logging.info(f"[utils.data.py] Final DataFrame shape: {new_df.shape}")
        return model_list, new_df


class RouterEvalLoader(BaseDatasetLoader):
    name = "RouterEval"
    hf_id = "linggm/Routereval"
    default_target_path = "./data/routerEval"

    def download(self) -> str:
        """Verify the pinned archive, then prepare the six Group A task files."""
        zip_path = download_asset(self.hf_id, "router_dataset.zip", self.target_path, self.source)
        required = [name for name in self.source["sha256"] if name.endswith(".pkl")]
        target = Path(self.target_path)
        missing = []
        for name in required:
            destination = target / name
            if destination.exists():
                verify_file(destination, self.source["sha256"][name])
            else:
                missing.append(name)
        if not missing:
            return self.target_path
        with tempfile.TemporaryDirectory(prefix=".routereval-", dir=target) as staging:
            with zipfile.ZipFile(zip_path) as archive:
                safe_extract_zip(archive, staging)
            extracted = Path(staging) / "router_dataset"
            for name in missing:
                path = extracted / name
                if not path.is_file():
                    raise FileNotFoundError(f"RouterEval archive is missing {name}")
                verify_file(path, self.source["sha256"][name])
            for name in missing:
                shutil.move(str(extracted / name), target / name)
        return self.target_path

    def process(self, dataset: Any) -> Any:
        """Load and normalize the six-task RouterEval Group A candidate pool."""
        group = self._group_a_candidates(dataset)
        models = group["common_models"]
        if not models:
            raise ValueError("RouterEval Group A has no common candidate models")
        frame = self._extract_performance_cost_data(group)
        if frame is None or frame.empty:
            raise ValueError("RouterEval Group A did not produce any aligned samples")
        self._validate_model_columns(frame, models)
        return models, self.normalize_cost_dataframe(frame)

    @staticmethod
    def _validate_model_columns(dataframe: pd.DataFrame, model_list) -> None:
        expected_performance = {f"model_{index}_performance" for index in range(len(model_list))}
        expected_cost = {f"model_{index}_cost" for index in range(len(model_list))}
        actual_performance = {
            column for column in dataframe.columns if re.fullmatch(r"model_\d+_performance", column)
        }
        actual_cost = {
            column for column in dataframe.columns if re.fullmatch(r"model_\d+_cost", column)
        }
        if actual_performance != expected_performance or actual_cost != expected_cost:
            raise ValueError(
                "RouterEval model columns do not match the canonical model list: "
                f"models={len(model_list)}, performance_columns={len(actual_performance)}, "
                f"cost_columns={len(actual_cost)}."
            )

    def _group_a_candidates(self, directory_path):
        """Find one canonical candidate order across the six evaluated tasks."""
        filenames = [
            "arc_router_dataset.pkl",
            "gsm8k_router_dataset.pkl",
            "harness_truthfulqa_mc_0_router_dataset.pkl",
            "hellaswag_router_dataset.pkl",
            "mmlu_router_dataset.pkl",
            "winogrande_router_dataset.pkl",
        ]
        model_sets = []
        for filename in filenames:
            file_path = Path(directory_path) / filename
            if not file_path.is_file():
                raise FileNotFoundError(f"RouterEval Group A is missing {filename}")
            data = load_verified_pickle(file_path, self.source["sha256"][filename])
            models = set()
            for difficulty in ("easy", "hard"):
                for pool in data.get(difficulty, {}).values():
                    for group in pool.values():
                        for name in group.get("model", []):
                            name = str(name).strip()
                            if name and self._extract_cost_from_name(name) is not None:
                                models.add(name)
            model_sets.append(models)
        return {"files": filenames, "common_models": sorted(set.intersection(*model_sets))}

    def _extract_performance_cost_data(self, df):
        """Align per-model utilities and costs across prompts and source pools."""
        group_info = df

        dataset_files = group_info.get("files", [])
        common_models = list(group_info.get("common_models", []))
        if len(common_models) != len(set(common_models)):
            raise ValueError("RouterEval common model list contains duplicate model names.")
        n_common = len(common_models)

        model_to_common_idx = {m: i for i, m in enumerate(common_models)}

        all_data = []

        for dataset_file in dataset_files:
            match = re.search(r"(.+)\.pkl$", dataset_file)
            dataset_name = match.group(1) if match else "unknown_dataset"
            file_path = os.path.join(self.target_path, dataset_file)

            data = load_verified_pickle(file_path, self.source["sha256"][dataset_file])

            all_samples = []
            sample_id_to_row = {}

            def _add_split_samples(split_type: str, idx_key: str, prompt_key: str):
                if "split_index" not in data or "prompt" not in data:
                    return
                if idx_key not in data["split_index"] or prompt_key not in data["prompt"]:
                    return

                indices = data["split_index"][idx_key]
                prompts = data["prompt"][prompt_key]
                if isinstance(indices, np.ndarray):
                    indices = indices.tolist()
                if isinstance(prompts, np.ndarray):
                    prompts = prompts.tolist()

                for idx, prompt in zip(indices, prompts):
                    clean_prompt = str(prompt).replace("\n", " ").replace("\r", " ")
                    clean_prompt = " ".join(clean_prompt.split())

                    sid = f"{dataset_name}{split_type}{idx}"
                    row_idx = len(all_samples)
                    sample_id_to_row[sid] = row_idx

                    all_samples.append(
                        {
                            "sample_id": sid,
                            "prompt": clean_prompt,
                            "eval_name": dataset_name,
                        }
                    )

            _add_split_samples("train", "train_indices", "train_prompt")
            _add_split_samples("val", "val_indices", "val_prompt")
            _add_split_samples("test", "test_indices", "test_prompt")

            if not all_samples:
                continue

            n_samples = len(all_samples)

            perf_mat = np.full((n_samples, n_common), np.nan, dtype=np.float32)
            cost_mat = np.full((n_samples, n_common), np.nan, dtype=np.float32)

            common_costs = np.array(
                [self._extract_cost_from_name(m) for m in common_models], dtype=np.float32
            )

            for difficulty in ["easy", "hard"]:
                if difficulty not in data:
                    continue

                diff_data = data[difficulty]
                for pool_size, pool_data in diff_data.items():
                    for _, group_data in pool_data.items():
                        if "model" not in group_data or "data" not in group_data:
                            continue

                        models = group_data["model"]
                        if isinstance(models, np.ndarray):
                            models = models.tolist()
                        models = [str(m) for m in models]

                        data_dict = group_data["data"]

                        local_to_common = []
                        for j, m in enumerate(models):
                            ci = model_to_common_idx.get(m, None)
                            if ci is not None:
                                local_to_common.append((j, ci))

                        if not local_to_common:
                            continue

                        local_cols = np.array([x[0] for x in local_to_common], dtype=np.int64)
                        common_cols = np.array([x[1] for x in local_to_common], dtype=np.int64)

                        for split_type in ["train", "val", "test"]:
                            score_key = f"{split_type}_score"
                            indices_key = f"{split_type}_indices"

                            if score_key not in data_dict:
                                continue
                            scores = data_dict[score_key]
                            if not isinstance(scores, np.ndarray) or scores.ndim != 2:
                                continue

                            if "split_index" not in data or indices_key not in data["split_index"]:
                                continue
                            sample_indices = data["split_index"][indices_key]
                            if isinstance(sample_indices, np.ndarray):
                                sample_indices = sample_indices.tolist()

                            row_idxs = []
                            valid_rows = []
                            for k, sid in enumerate(sample_indices):
                                full_sid = f"{dataset_name}{split_type}{sid}"
                                ri = sample_id_to_row.get(full_sid, None)
                                if ri is not None:
                                    row_idxs.append(ri)
                                    valid_rows.append(k)

                            if not row_idxs:
                                continue

                            row_idxs = np.array(row_idxs, dtype=np.int64)
                            valid_rows = np.array(valid_rows, dtype=np.int64)

                            scores_sub = scores[valid_rows[:, None], local_cols[None, :]].astype(
                                np.float32, copy=False
                            )
                            perf_mat[row_idxs[:, None], common_cols[None, :]] = scores_sub

                            cost_mat[row_idxs[:, None], common_cols[None, :]] = common_costs[
                                common_cols
                            ][None, :]

            perf_cols = [f"model_{i}_performance" for i in range(n_common)]
            cost_cols = [f"model_{i}_cost" for i in range(n_common)]

            base_df = pd.DataFrame(all_samples)
            perf_df = pd.DataFrame(perf_mat, columns=perf_cols)
            cost_df = pd.DataFrame(cost_mat, columns=cost_cols)

            data_df = pd.concat([base_df, perf_df, cost_df], axis=1).copy()

            all_data.append(data_df)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    def _extract_cost_from_name(self, model_name):
        "Extract model size, supporting formats like -3b and 3x5b"

        model_name = model_name.lower()

        mult_match = re.search(r"-(\d*\.?\d+(?:[x×]\d*\.?\d+)+)b", model_name)
        if mult_match:
            expr = mult_match.group(1)
            nums = [float(n) if "." in n else int(n) for n in re.split(r"\s*[x×]\s*", expr)]
            result = 1
            for n in nums:
                result *= n
            return int(result)

        num_match = re.search(r"-(\d+(?:\.\d+)?)\s*b\b", model_name)
        if num_match:
            num = num_match.group(1)
            return float(num) if "." in num else int(num)

        return None


# Registry to allow users to add new dataset loaders
_LOADER_REGISTRY: dict[str, type[BaseDatasetLoader]] = {
    RouterbenchLoader.name: RouterbenchLoader,
    MixinstructLoader.name: MixinstructLoader,
    MMRBenchLoader.name: MMRBenchLoader,
    RouterEvalLoader.name: RouterEvalLoader,
}


def register_loader(name: str, loader_cls: type[BaseDatasetLoader]) -> None:
    """
    Register a new loader class under `name`. Overwrites existing entry if present.
    """
    _LOADER_REGISTRY[name] = loader_cls


def get_loader(name: str) -> type[BaseDatasetLoader]:
    for registered, loader in _LOADER_REGISTRY.items():
        if registered.casefold() == name.casefold():
            return loader
    raise ValueError(f"Unknown dataset: {name!r}. Valid options: {list(_LOADER_REGISTRY)}")


def download_dataset(args: Any) -> tuple[pd.DataFrame, pd.DataFrame, Any | None]:
    """
    Select loader based on args["dataset"]["name"] and run its pipeline.
    Uses args["dataset"]["dataset_dir"] as target_path.
    Returns: (train_df, test_df, model_list_or_None)
    """
    dataset_name = args["dataset"]["name"]
    LoaderCls = get_loader(dataset_name)
    target_path = args["dataset"]["dataset_dir"]

    loader = LoaderCls(target_path=target_path, args=args)
    return loader.run()
