"""Small real training loops and shared evaluation, without remote model calls."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from main import load_config
from methods import EquiRouter
from methods.system_one import SystemOneRouter
from utils.metrics import evaluate_budget_curve


@pytest.fixture
def training_data(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    train = pd.DataFrame(
        {
            "prompt": ["north", "south", "east", "west"],
            "model_0_performance": [1.0, 0.0, 1.0, 0.0],
            "model_1_performance": [0.0, 1.0, 0.0, 1.0],
            "model_0_cost": [2.0] * 4,
            "model_1_cost": [1.0] * 4,
        }
    )
    test = train.iloc[:2].copy()
    test["prompt"] = ["new north", "new south"]
    train_x = np.array([[1, 0], [0, 1], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    test_x = train_x[:2].copy()

    class Encoder:
        def run_embed(self, texts=None, images=None):
            return train_x if len(texts) == 4 else test_x

    monkeypatch.setattr(
        "methods.base.download_dataset", lambda args: (train.copy(), test.copy(), ["a", "b"])
    )
    monkeypatch.setattr("methods.base.Embedder", lambda args: Encoder())
    args = load_config("mixinstruct", device="cpu")
    args.update(cache_embeddings=False, system_one={"neighbors": 2})
    args["embeddings"]["out_dim"] = 2
    args["perf_model"].update(trunk_hidden=[8], trunk_out_dim=6, model_emb_dim=3, head_hidden=[6])
    args["model"].update(hidden_sizes=[4], dataset_overrides={})
    args["training"].update(epochs=2, batch_size=2, lr=0.01, dataset_overrides={})
    return args, train, test, train_x, test_x


def test_equirouter_real_training_predictions_and_evaluation(training_data):
    args, _, _, _, test_x = training_data
    router = EquiRouter(args)
    with pytest.raises(RuntimeError, match="train"):
        router.predict(test_x)
    router.train()
    scores, costs = router.predict(test_x)
    assert scores.shape == costs.shape == (2, 2)
    assert np.isfinite(scores).all() and np.isfinite(costs).all()
    result = router.evaluate()
    assert len(result["curve"]) == 100
    assert np.isfinite(result["normalized_auc"])
    empty, empty_costs = router.predict(np.empty((0, 2), dtype=np.float32))
    assert empty.shape == empty_costs.shape == (0, 2)


class RecordingBackend:
    model = "offline-test-model"
    cache_identity = {"model": model, "version": "1"}

    def __init__(self):
        self.requests = []

    def predict(self, request):
        self.requests.append(copy.deepcopy(request))
        return {
            "answers": {
                "selected_model": {
                    "choice": "model_1",
                    "probabilities": {"model_1": 0.8, "model_0": 0.2},
                }
            }
        }


@pytest.mark.parametrize("multimodal", [False, True])
def test_system_one_integration_is_training_only_and_reuses_budget_scores(
    training_data, tmp_path, multimodal
):
    args, train, test, _, test_x = training_data
    args["method"] = "Jev"
    if multimodal:
        args["modality"] = "text+image"
        paths = []
        for index in range(6):
            path = tmp_path / f"sample-{index}.png"
            Image.new("RGB", (32, 32), (index * 30, 0, 0)).save(path)
            paths.append(str(path))
        train["image_path"], test["image_path"] = paths[:4], paths[4:]
    backend = RecordingBackend()
    router = SystemOneRouter(args, backend=backend)
    with pytest.raises(RuntimeError, match="train"):
        router.predict(test_x)
    router.train()
    scores, costs = router.predict(test_x)
    assert len(backend.requests) == 2
    np.testing.assert_allclose(scores, [[0.2, 0.8], [0.2, 0.8]])
    np.testing.assert_allclose(costs, [[2.0, 1.0], [2.0, 1.0]])
    for payload in backend.requests:
        history = payload["state"]["shared_training_history"]
        assert len(history) == 2
        assert all(entry["query"] in train["prompt"].tolist() for entry in history)
        assert str(tmp_path) not in json.dumps(payload)
        assert "_cost" not in json.dumps(payload)
        assert ("images" in payload) == multimodal
        assert len(payload.get("images", [])) == (3 if multimodal else 0)
    # Test labels/costs cannot affect the request or cached utility predictions.
    router.test_df["model_0_performance"] = 12345
    router.test_df["model_1_cost"] = 99999
    repeated, _ = router.predict(test_x)
    np.testing.assert_array_equal(repeated, scores)
    assert len(backend.requests) == 2
    result = router.evaluate()
    assert len(result["curve"]) == 100 and len(backend.requests) == 2


def test_budget_evaluation_uses_choice_ranking_after_cost_filtering():
    result = evaluate_budget_curve(
        [[0.9, 0.1]], [[2.0, 1.0]], [[1.0, 0.75]], [[2.0, 1.0]], budget_points=2
    )
    assert result["curve"][0]["performance"] == 0.75
    assert result["curve"][1]["performance"] == 1.0


def test_config_loading_works_outside_project_and_selects_system_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = load_config("ROUTERBENCH", method="jev")
    assert args["method"] == "Jev" and args["system_one"]["neighbors"] == 32
    assert load_config("mmrbench", method="clef-flash")["method"] == "ClefFlash"
