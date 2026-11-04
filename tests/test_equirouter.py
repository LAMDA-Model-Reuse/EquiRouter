import json
from pathlib import Path

import torch
import torch.nn.functional as F

from methods.EquiRouter import (
    BaseNet,
    pairwise_rank_loss,
    resolve_dataset_config,
)


def test_strict_utility_pair_has_correct_gradient_direction():
    scores = torch.zeros((1, 2), requires_grad=True)
    loss = pairwise_rank_loss(scores, torch.tensor([[1.0, 0.0]]), tie_cost_weight=0)
    loss.backward()
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] > 0


def test_caro_cost_tie_prefers_the_cheaper_model():
    scores = torch.zeros((1, 2), requires_grad=True)
    loss = pairwise_rank_loss(
        scores,
        torch.ones((1, 2)),
        costs=torch.tensor([[1.0, 2.0]]),
    )
    loss.backward()
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] > 0


def test_generalized_caro_uses_cost_within_indifference_radius():
    scores = torch.zeros((1, 2), requires_grad=True)
    loss = pairwise_rank_loss(
        scores,
        torch.tensor([[0.795, 0.800]]),
        costs=torch.tensor([[1.0, 2.0]]),
        tie_epsilon=0.01,
    )
    loss.backward()
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] > 0


def test_caro_normalizes_each_query_before_batch_average():
    scores = torch.tensor([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    loss = pairwise_rank_loss(scores, labels, tie_cost_weight=0)
    first = F.softplus(torch.tensor(-2.0))
    second = torch.stack(
        [
            F.softplus(torch.tensor(-1.0)),
            F.softplus(torch.tensor(-2.0)),
            F.softplus(torch.tensor(-1.0)),
        ]
    ).mean()
    assert torch.allclose(loss, (first + second) / 2)


def test_qmir_output_shape_and_identity_initialization():
    network = BaseNet(
        in_dim=4,
        num_models=3,
        trunk_hidden=(8,),
        trunk_out_dim=6,
        model_emb_dim=5,
        head_hidden=(7,),
        film_identity_init=True,
    )
    assert network(torch.randn(2, 4)).shape == (2, 3)
    assert torch.allclose(network.film_proj.bias[:6], torch.ones(6))
    assert torch.allclose(network.film_proj.bias[6:], torch.zeros(6))


def test_released_configuration_keeps_caro_in_every_profile():
    config = json.loads(Path("configs/routers/EquiRouter.json").read_text(encoding="utf-8"))
    assert config["training"]["require_caro"] is True
    assert config["training"]["rank_weight"] > 0
    assert config["training"]["tie_cost_weight"] > 0
    for dataset in config["training"]["dataset_overrides"]:
        profile = resolve_dataset_config(config["training"], dataset)
        assert profile["rank_weight"] > 0
        assert profile["tie_cost_weight"] > 0
