"""Minimal shared runtime used by the released EquiRouter implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.data import download_dataset
from utils.embedding import Embedder
from utils.metrics import evaluate_budget_curve
from utils.sources import file_sha256


class BaseRouter(ABC):
    """Dataset, embedding, reproducibility, and evaluation support."""

    @staticmethod
    def _resolve_device(device_arg: str | None) -> torch.device:
        if device_arg is None or str(device_arg).casefold() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested = torch.device(device_arg)
        if requested.type != "cuda":
            return requested
        if not torch.cuda.is_available():
            logging.warning("CUDA is unavailable; using CPU instead of %s.", requested)
            return torch.device("cpu")
        if requested.index is not None and not 0 <= requested.index < torch.cuda.device_count():
            logging.warning("CUDA device %s is not visible; using CPU instead.", requested)
            return torch.device("cpu")
        return requested

    def __init__(self, args: dict[str, Any]) -> None:
        self.args = deepcopy(args)
        self.train_df, self.test_df, self.model_list = download_dataset(self.args)
        configured_device = self.args.get("device", "auto")
        self.device = self._resolve_device(configured_device)
        self.args["device"] = str(self.device)
        self.args["embeddings"]["device"] = str(self.device)
        self.embedder = Embedder(self.args)
        self.seed = int(self.args["seed"])
        self._training_cost_bounds: tuple[np.ndarray, np.ndarray] | None = None

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @abstractmethod
    def train(self) -> None:
        """Fit the router on the configured training partition."""

    @abstractmethod
    def predict(self, test_embedding):
        """Return predicted utility and cost matrices with shape ``(N, M)``."""

    def _target_matrices(self, frame) -> tuple[np.ndarray, np.ndarray]:
        count = len(self.model_list)
        performance = frame[[f"model_{index}_performance" for index in range(count)]].to_numpy(
            dtype=np.float32
        )
        cost = frame[[f"model_{index}_cost" for index in range(count)]].to_numpy(dtype=np.float32)
        return performance, cost

    def evaluate(self) -> dict[str, Any]:
        embeddings = self._prepare_embeddings(self.test_df, "test")
        predicted_performance, predicted_cost = self.predict(embeddings)
        predicted_cost = self._clip_predicted_costs(predicted_cost)
        true_performance, true_cost = self._target_matrices(self.test_df)
        budget_points = int(self.args.get("paper_reproduction", {}).get("budget_points", 100))
        metrics = evaluate_budget_curve(
            predicted_performance,
            predicted_cost,
            true_performance,
            true_cost,
            model_names=self.model_list,
            budget_points=budget_points,
        )
        output_dir = Path("outputs") / self.args["dataset"]["name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.args.get('method', 'EquiRouter')}_{time.time_ns()}.json"
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logging.info(
            "Peak=%.6f QNC=%s nAUC=%.6f; metrics saved to %s",
            metrics["peak"],
            metrics["qnc"],
            metrics["normalized_auc"],
            output_path,
        )
        return metrics

    def _clip_predicted_costs(self, predicted_cost) -> np.ndarray:
        values = np.asarray(predicted_cost, dtype=np.float32)
        if self._training_cost_bounds is None:
            return values
        lower, upper = self._training_cost_bounds
        if values.ndim != 2 or values.shape[1] != len(lower):
            raise ValueError(
                f"Predicted cost must have shape (N, {len(lower)}), got {values.shape}."
            )
        return np.where(np.isfinite(values), np.clip(values, lower, upper), values)

    def _prepare_embeddings(self, frame, split_name: str) -> np.ndarray:
        modalities = self.args["modality"].split("+")
        texts = frame["prompt"].astype(str).tolist() if "text" in modalities else None
        images = frame["image_path"].astype(str).tolist() if "image" in modalities else None
        embedding_config = self.args["embeddings"]
        use_cache = bool(self.args.get("cache_embeddings", False))
        cache_path: Path | None = None
        if use_cache:
            digest = hashlib.sha256()
            metadata = {
                "version": 3,
                "split": split_name,
                "modality": self.args["modality"],
                "text_model": embedding_config.get("text_model"),
                "text_model_dir": embedding_config.get("text_model_dir"),
                "image_model": embedding_config.get("image_model"),
                "normalize": embedding_config.get("normalize", False),
                "out_dim": embedding_config.get("out_dim"),
                "rows": len(frame),
                "encoder_identity": getattr(self.embedder, "cache_identity", {}),
            }
            digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
            for values in (texts, images):
                for value in values or ():
                    digest.update(value.encode("utf-8", errors="surrogatepass"))
                    digest.update(b"\0")
            for image_path in images or ():
                digest.update(file_sha256(image_path).encode())
            cache_dir = Path(self.args["dataset"]["dataset_dir"]) / ".embedding_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{split_name}_{digest.hexdigest()[:20]}.npy"
            if cache_path.is_file():
                logging.info("Loading embedding cache: %s", cache_path)
                return self._validate_embeddings(
                    np.load(cache_path, allow_pickle=False), len(frame)
                )

        embeddings = self.embedder.run_embed(texts=texts, images=images)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        result = self._validate_embeddings(embeddings, len(frame))
        if cache_path is not None:
            temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npy")
            np.save(temporary, result, allow_pickle=False)
            os.replace(temporary, cache_path)
            logging.info("Saved embedding cache: %s", cache_path)
        return result

    def _validate_embeddings(self, embeddings, rows: int) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        expected = (rows, int(self.args["embeddings"]["out_dim"]))
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"Expected finite embeddings of shape {expected}, got {values.shape}")
        return values

    def _prepare_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embeddings = self._prepare_embeddings(self.train_df, "train")
        performance, cost = self._target_matrices(self.train_df)
        if not np.isfinite(cost).any(axis=0).all():
            missing = np.flatnonzero(~np.isfinite(cost).any(axis=0)).tolist()
            raise ValueError(f"No finite training cost for model indices {missing}.")
        self._training_cost_bounds = (
            np.nanmin(cost, axis=0).astype(np.float32),
            np.nanmax(cost, axis=0).astype(np.float32),
        )
        return embeddings, performance, cost
