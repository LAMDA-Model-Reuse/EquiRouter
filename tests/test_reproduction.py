import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from main import load_config
from methods.EquiRouter import resolve_dataset_config
from scripts.reproduce_equirouter import DATASETS, run


@pytest.mark.parametrize("dataset", DATASETS)
def test_reproduction_trains_one_router_and_evaluates_its_predictions(dataset, tmp_path):
    embeddings = np.zeros((2, 4), dtype=np.float32)
    scores = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    costs = np.array([[2.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "model_0_performance": scores[:, 0],
            "model_1_performance": scores[:, 1],
            "model_0_cost": costs[:, 0],
            "model_1_cost": costs[:, 1],
        }
    )
    routers = []

    def make_router(config):
        router = MagicMock()
        router.args = config
        router.seed = config["seed"]
        router.model_list = ["first", "second"]
        router.test_df = frame
        router._prepare_embeddings.return_value = embeddings
        router._target_matrices.return_value = (scores, costs)
        router.predict.return_value = (scores, costs)
        router._clip_predicted_costs.return_value = costs
        routers.append(router)
        return router

    with (
        patch("scripts.reproduce_equirouter.EquiRouter", side_effect=make_router) as constructor,
        patch("scripts.reproduce_equirouter.setup_logging"),
    ):
        result = run(dataset, "cpu", tmp_path)

    constructor.assert_called_once()
    routers[0].train.assert_called_once_with()
    routers[0].predict.assert_called_once_with(embeddings)
    routers[0]._clip_predicted_costs.assert_called_once_with(costs)
    config = constructor.call_args.args[0]
    assert result["configuration"] == resolve_dataset_config(config["training"], DATASETS[dataset])
    assert result["metrics"]["peak"] == 1.0
    assert len(result["metrics"]["curve"]) == 100
    assert json.loads((tmp_path / f"{dataset}.json").read_text()) == result


def test_qmir_architecture_is_shared_across_benchmarks():
    profiles = [
        resolve_dataset_config(load_config(name)["perf_model"], DATASETS[name]) for name in DATASETS
    ]
    assert all(profile == profiles[0] for profile in profiles)


def test_mmrbench_defaults_use_declared_training_profile():
    config = load_config("mmrbench")
    performance = resolve_dataset_config(config["perf_model"], "MMRBench")
    training = resolve_dataset_config(config["training"], "MMRBench")
    assert performance["film_identity_init"] is False
    assert config["seed"] == 42
    expected = {
        "lr": 0.0015,
        "epochs": 40,
        "batch_size": 256,
        "optimizer": "adam",
        "pointwise_weight": 33.0,
        "rank_weight": 0.15,
        "tie_cost_weight": 1.0,
        "weight_decay": 0.01,
        "grad_clip": 5.0,
        "require_caro": True,
        "tie_epsilon": 0.0,
    }
    for name, value in expected.items():
        assert training[name] == value


def test_benchmark_profiles_only_override_training_hyperparameters():
    training = load_config("mmrbench")["training"]
    defaults = set(training) - {"dataset_overrides"}
    for profile in training["dataset_overrides"].values():
        assert set(profile) <= defaults
