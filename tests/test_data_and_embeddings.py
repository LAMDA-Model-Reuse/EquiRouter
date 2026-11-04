import hashlib
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from methods.base import BaseRouter
from utils.data import BaseDatasetLoader, RouterbenchLoader, RouterEvalLoader, get_loader
from utils.embedding import _ImageDataset
from utils.metrics import evaluate_budget_curve


def test_grouped_split_preserves_tasks_and_seeded_partition():
    args = {
        "seed": 42,
        "dataset": {"split": {"mode": "out-of-domain", "ratios": {"train": 0.4, "test": 0.6}}},
    }
    loader = BaseDatasetLoader(None, args)
    frame = pd.DataFrame({"eval_name": np.repeat(list("abcdef"), 2), "id": np.arange(12)})
    train, test = loader.split(frame)
    assert train["eval_name"].unique().tolist() == ["a", "b"]
    assert set(train["eval_name"]).isdisjoint(test["eval_name"])
    assert sorted([*train["id"], *test["id"]]) == list(range(12))
    assert get_loader("ROUTERBENCH") is RouterbenchLoader


def test_equal_cost_normalization_is_finite_without_runtime_warnings():
    frame = pd.DataFrame({"model_0_cost": [2.0, 4.0], "model_1_cost": [2.0, 4.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        normalized = BaseDatasetLoader(None, {}).normalize_cost_dataframe(frame)
    np.testing.assert_allclose(normalized, 0.5)


def test_routerbench_accepts_cached_outcomes_without_response_text(tmp_path):
    data = [
        {
            "sample_id": 0,
            "model_name": "gpt-4-1106-preview",
            "prompt": "a query",
            "performance": 1.0,
            "cost": 2.0,
        }
    ]
    path = tmp_path / "routerbench_raw.pkl"
    path.write_bytes(pickle.dumps(data))
    loader = RouterbenchLoader(str(tmp_path), {})
    loader.source["sha256"]["routerbench_raw.pkl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    models, frame = loader.process(str(path))
    index = models.index("gpt-4-1106-preview")
    assert frame[f"model_{index}_performance"].iloc[0] == 1.0
    assert frame[f"model_{index}_cost"].iloc[0] == 2.0


def test_routereval_rejects_incomplete_task_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="Group A is missing"):
        RouterEvalLoader(str(tmp_path), {}).process(str(tmp_path))


def test_missing_image_does_not_silently_create_fake_features(tmp_path):
    dataset = _ImageDataset([tmp_path / "missing.png"], lambda image: image)
    with pytest.raises(ValueError, match="index 0"):
        dataset[0]


class StubRouter(BaseRouter):
    def train(self):
        return None

    def predict(self, test_embedding):
        return test_embedding, test_embedding


def test_embedding_cache_invalidates_changed_image_content(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "red").save(image_path)
    router = object.__new__(StubRouter)
    router.args = {
        "modality": "text+image",
        "cache_embeddings": True,
        "dataset": {"dataset_dir": str(tmp_path)},
        "embeddings": {"out_dim": 2},
    }

    class Encoder:
        calls = 0

        def run_embed(self, **kwargs):
            self.calls += 1
            return np.full((1, 2), self.calls, dtype=np.float32)

    router.embedder = Encoder()
    frame = pd.DataFrame({"prompt": ["a"], "image_path": [str(image_path)]})
    first = router._prepare_embeddings(frame, "test")
    np.testing.assert_array_equal(router._prepare_embeddings(frame, "test"), first)
    assert router.embedder.calls == 1
    Image.new("RGB", (20, 20), "blue").save(image_path)
    assert not np.array_equal(router._prepare_embeddings(frame, "test"), first)
    assert router.embedder.calls == 2


def test_embedding_cache_uses_content_even_if_image_stat_is_unchanged(tmp_path):
    image = tmp_path / "image.raw"
    image.write_bytes(b"aaaa")
    stat = image.stat()
    router = object.__new__(StubRouter)
    router.args = {
        "modality": "text+image",
        "cache_embeddings": True,
        "dataset": {"dataset_dir": str(tmp_path)},
        "embeddings": {"out_dim": 2},
    }

    class Encoder:
        cache_identity = {"revision": "first"}
        calls = 0

        def run_embed(self, **kwargs):
            self.calls += 1
            return np.full((1, 2), self.calls, dtype=np.float32)

    router.embedder = Encoder()
    frame = pd.DataFrame({"prompt": ["query"], "image_path": [str(image)]})
    first = router._prepare_embeddings(frame, "test")
    image.write_bytes(b"bbbb")
    os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = router._prepare_embeddings(frame, "test")
    assert not np.array_equal(first, second)
    router.embedder.cache_identity = {"revision": "second"}
    third = router._prepare_embeddings(frame, "test")
    assert not np.array_equal(second, third)


def test_zero_cost_target_has_undefined_qnc_instead_of_division_error():
    result = evaluate_budget_curve([[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]])
    assert result["qnc"] is None and result["cost_saving_percent_at_target"] is None


def test_multimodal_encoder_loads_one_shared_clip_model(monkeypatch):
    from unittest.mock import Mock

    import clip
    import torch

    from main import load_config
    from utils.embedding import Embedder

    network = Mock()
    network.encode_text.return_value = torch.ones((1, 512))
    network.encode_image.return_value = torch.ones((1, 512))
    load = Mock(return_value=(network, lambda image: torch.zeros((3, 224, 224))))
    monkeypatch.setattr(clip, "load", load)
    monkeypatch.setattr(
        clip, "tokenize", lambda values, **kwargs: torch.zeros((len(values), 77), dtype=torch.long)
    )
    args = load_config("mmrbench", device="cpu")
    args["embeddings"].update(device="cpu", image_workers=0)
    encoder = Embedder(args)
    result = encoder.run_embed(texts=["query"], images=[Image.new("RGB", (32, 32))])
    assert result.shape == (1, 1024)
    assert torch.isfinite(result).all()
    load.assert_called_once()
    network.encode_text.assert_called_once()
    network.encode_image.assert_called_once()


@pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
def test_evaluation_rejects_empty_partitions(shape):
    with pytest.raises(ValueError, match="requires queries and candidate"):
        evaluate_budget_curve(*[np.empty(shape)] * 4)
