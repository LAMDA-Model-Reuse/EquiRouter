import hashlib
import io
import json
import pickle
import stat
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from main import load_config
from methods.base import BaseRouter
from utils.data import RouterEvalLoader, safe_extract_tar, safe_extract_zip


class _ConcreteRouter(BaseRouter):
    def train(self):
        return None

    def predict(self, test_embedding):
        return test_embedding, test_embedding


class RouterEvalIdentityTest(unittest.TestCase):
    GROUP_A_FILES = (
        "arc_router_dataset.pkl",
        "gsm8k_router_dataset.pkl",
        "harness_truthfulqa_mc_0_router_dataset.pkl",
        "hellaswag_router_dataset.pkl",
        "mmlu_router_dataset.pkl",
        "winogrande_router_dataset.pkl",
    )

    def test_model_list_and_columns_share_one_canonical_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            payload = {
                "split_index": {"train_indices": [0]},
                "prompt": {"train_prompt": ["question"]},
                "easy": {
                    "2": {
                        "pool": {
                            "model": ["zeta-2b", "alpha-1b"],
                            "data": {"train_score": np.asarray([[0.9, 0.1]])},
                        }
                    }
                },
            }
            for filename in self.GROUP_A_FILES:
                with (Path(temporary_directory) / filename).open("wb") as handle:
                    pickle.dump(payload, handle)
            loader = RouterEvalLoader(temporary_directory, {"seed": 0})
            loader.source["sha256"].update(
                {
                    name: hashlib.sha256(
                        (Path(temporary_directory) / name).read_bytes()
                    ).hexdigest()
                    for name in self.GROUP_A_FILES
                }
            )
            model_list, dataframe = loader.process(temporary_directory)
            self.assertEqual(model_list, ["alpha-1b", "zeta-2b"])
            np.testing.assert_allclose(dataframe["model_0_performance"], 0.1)
            np.testing.assert_allclose(dataframe["model_1_performance"], 0.9)


class SafeArchiveTest(unittest.TestCase):
    def test_tar_rejects_parent_traversal_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "bad.tar.gz"
            output_path = Path(temporary_directory) / "output"
            with tarfile.open(archive_path, "w:gz") as archive:
                safe = tarfile.TarInfo("safe.txt")
                safe.size = 4
                archive.addfile(safe, io.BytesIO(b"safe"))
                unsafe = tarfile.TarInfo("../escape.txt")
                unsafe.size = 6
                archive.addfile(unsafe, io.BytesIO(b"escape"))
            with tarfile.open(archive_path, "r:gz") as archive:
                with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                    safe_extract_tar(archive, str(output_path))
            self.assertFalse((Path(temporary_directory) / "escape.txt").exists())
            self.assertFalse((output_path / "safe.txt").exists())

    def test_tar_rejects_links(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "link.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo("link")
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                archive.addfile(info)
            with tarfile.open(archive_path, "r:gz") as archive:
                with self.assertRaisesRegex(ValueError, "links and special files"):
                    safe_extract_tar(archive, str(Path(temporary_directory) / "output"))

    def test_zip_rejects_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            traversal = Path(temporary_directory) / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as archive:
                archive.writestr("../escape.txt", "escape")
            with zipfile.ZipFile(traversal) as archive:
                with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                    safe_extract_zip(archive, str(Path(temporary_directory) / "output"))

            symlink = Path(temporary_directory) / "symlink.zip"
            with zipfile.ZipFile(symlink, "w") as archive:
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, "target")
            with zipfile.ZipFile(symlink) as archive:
                with self.assertRaisesRegex(ValueError, "symlink"):
                    safe_extract_zip(archive, str(Path(temporary_directory) / "output"))

    def test_valid_archives_extract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "output"
            tar_path = Path(temporary_directory) / "valid.tar.gz"
            with tarfile.open(tar_path, "w:gz") as archive:
                info = tarfile.TarInfo("nested/tar.txt")
                info.size = 3
                archive.addfile(info, io.BytesIO(b"tar"))
            with tarfile.open(tar_path, "r:gz") as archive:
                safe_extract_tar(archive, str(target))
            zip_path = Path(temporary_directory) / "valid.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("nested/zip.txt", "zip")
            with zipfile.ZipFile(zip_path) as archive:
                safe_extract_zip(archive, str(target))
            self.assertEqual((target / "nested/tar.txt").read_text(), "tar")
            self.assertEqual((target / "nested/zip.txt").read_text(), "zip")


class ConfigurationTest(unittest.TestCase):
    def test_dataset_name_and_device_override_are_case_insensitive(self):
        config = load_config("MIXINSTRUCT", device="cpu")
        self.assertEqual(config["method"], "EquiRouter")
        self.assertEqual(config["device"], "cpu")

    def test_only_four_benchmarks_and_one_router_config_are_published(self):
        benchmarks = sorted(path.stem for path in Path("configs/benchmarks").glob("*.json"))
        routers = sorted(path.stem for path in Path("configs/routers").glob("*.json"))
        self.assertEqual(benchmarks, ["mixinstruct", "mmrbench", "routerbench", "routereval"])
        self.assertEqual(routers, ["EquiRouter"])
        for path in Path("configs").rglob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))

    def test_unavailable_cuda_index_falls_back_to_cpu(self):
        with (
            patch("methods.base.torch.cuda.is_available", return_value=True),
            patch("methods.base.torch.cuda.device_count", return_value=1),
        ):
            self.assertEqual(str(BaseRouter._resolve_device("cuda:3")), "cpu")

    def test_router_and_embedding_share_resolved_device(self):
        args = {
            "device": "cpu",
            "embeddings": {},
            "seed": 7,
        }
        with (
            patch("methods.base.download_dataset", return_value=(object(), object(), [])),
            patch("methods.base.Embedder", return_value=object()),
        ):
            router = _ConcreteRouter(args)
        self.assertEqual(str(router.device), "cpu")
        self.assertEqual(router.args["embeddings"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
