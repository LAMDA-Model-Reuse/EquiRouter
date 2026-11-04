"""Official EquiRouter implementation.

EquiRouter combines a query-model interaction representation (QMIR) with the
cost-aware ranking objective (CARO). CARO keeps strict utility pairs and
lower-cost preferences between utility-indifferent models. The training
objective combines pointwise MSE with per-query normalized CARO.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from methods.base import BaseRouter
from utils.build_model import init_model
from utils.cost import fit_cost_model, make_optimizer

logger = logging.getLogger(__name__)


def resolve_dataset_config(config: dict, dataset_name: str) -> dict:
    """Return a config section with its case-insensitive dataset override."""
    resolved = dict(config)
    overrides = resolved.pop("dataset_overrides", {})
    for name, values in overrides.items():
        if str(name).casefold() == dataset_name.casefold():
            resolved.update(values)
            break
    return resolved


class BaseNet(nn.Module):
    """QMIR performance network producing one score per candidate model."""

    def __init__(
        self,
        in_dim: int,
        num_models: int,
        trunk_hidden: Sequence[int] = (2048, 1024),
        trunk_out_dim: int = 512,
        model_emb_dim: int = 64,
        head_hidden: Sequence[int] = (512,),
        use_layernorm: bool = True,
        dropout: float = 0.0,
        film_identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.num_models = int(num_models)
        self.trunk_out_dim = int(trunk_out_dim)
        self.model_emb_dim = int(model_emb_dim)

        trunk_layers: list[nn.Module] = []
        previous = int(in_dim)
        for width in trunk_hidden:
            width = int(width)
            trunk_layers.extend((nn.Linear(previous, width), nn.GELU()))
            if use_layernorm:
                trunk_layers.append(nn.LayerNorm(width))
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
            previous = width
        trunk_layers.extend((nn.Linear(previous, self.trunk_out_dim), nn.GELU()))
        if use_layernorm:
            trunk_layers.append(nn.LayerNorm(self.trunk_out_dim))
        self.trunk = nn.Sequential(*trunk_layers)

        self.model_embeddings = nn.Parameter(
            torch.randn(self.num_models, self.model_emb_dim) * 0.02
        )
        self.film_proj = nn.Linear(self.model_emb_dim, self.trunk_out_dim * 2)
        self.model_proj = nn.Linear(self.model_emb_dim, self.trunk_out_dim)

        head_layers: list[nn.Module] = []
        previous = self.trunk_out_dim * 4
        for width in head_hidden:
            width = int(width)
            head_layers.extend((nn.Linear(previous, width), nn.GELU(), nn.LayerNorm(width)))
            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))
            previous = width
        head_layers.append(nn.Linear(previous, 1))
        self.head = nn.Sequential(*head_layers)

        self._init_weights()
        if film_identity_init:
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)
            with torch.no_grad():
                self.film_proj.bias[: self.trunk_out_dim].fill_(1.0)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.trunk(inputs)
        batch_size = query.shape[0]
        model_features = self.model_proj(self.model_embeddings).unsqueeze(0)
        expanded_query = query.unsqueeze(1).expand(-1, self.num_models, -1)
        film = self.film_proj(self.model_embeddings).view(self.num_models, 2, self.trunk_out_dim)
        gamma = film[:, 0].unsqueeze(0)
        beta = film[:, 1].unsqueeze(0)
        conditioned_query = gamma * expanded_query + beta
        expanded_model = model_features.expand(batch_size, -1, -1)
        interaction = torch.cat(
            (
                conditioned_query,
                expanded_model,
                conditioned_query * expanded_model,
                torch.abs(conditioned_query - expanded_model),
            ),
            dim=-1,
        )
        return self.head(interaction.flatten(0, 1)).view(batch_size, self.num_models)


class EquiRouter(BaseRouter):
    """QMIR router with configurable CARO and pointwise MSE supervision."""

    def __init__(self, args: dict) -> None:
        super().__init__(args)
        dataset_name = str(self.args.get("dataset", {}).get("name", ""))
        input_dim = int(self.args["embeddings"]["out_dim"])
        model_count = len(self.model_list)

        cost_config = resolve_dataset_config(self.args.get("model", {}), dataset_name)
        cost_args = dict(self.args)
        cost_args["model"] = cost_config
        self.model_cost = init_model(cost_args, input_dim=input_dim, out_dim=model_count)

        performance_config = resolve_dataset_config(self.args.get("perf_model", {}), dataset_name)
        self.model_perf = BaseNet(
            in_dim=input_dim,
            num_models=model_count,
            trunk_hidden=performance_config.get("trunk_hidden", (2048, 1024)),
            trunk_out_dim=int(performance_config.get("trunk_out_dim", 512)),
            model_emb_dim=int(performance_config.get("model_emb_dim", 64)),
            head_hidden=performance_config.get("head_hidden", (512,)),
            use_layernorm=bool(performance_config.get("use_layernorm", True)),
            dropout=float(performance_config.get("dropout", 0.0)),
            film_identity_init=bool(performance_config.get("film_identity_init", True)),
        )
        self.model_cost.to(self.device)
        self.model_perf.to(self.device)

    def train(self) -> None:
        embeddings, performance, costs = self._prepare_training_data()
        dataset_name = str(self.args.get("dataset", {}).get("name", ""))
        config = resolve_dataset_config(self.args.get("training", {}), dataset_name)

        epochs = int(config.get("epochs", 10))
        batch_size = int(config.get("batch_size", 256))
        gradient_clip = float(config.get("grad_clip", 0.0))
        pointwise_weight = float(config.get("pointwise_weight", 1.0))
        caro_weight = float(config.get("rank_weight", 1.0))
        tie_epsilon = float(config.get("tie_epsilon", 0.0))
        tie_cost_weight = float(config.get("tie_cost_weight", 1.0))
        temperature = float(config.get("rank_temperature", 1.0))
        margin = float(config.get("rank_margin", 0.0))
        require_caro = bool(config.get("require_caro", True))

        if str(config.get("loss", "mse")).casefold() != "mse":
            raise ValueError("EquiRouter supports only MSE as its pointwise loss")
        if require_caro and (caro_weight <= 0 or tie_cost_weight <= 0):
            raise ValueError("CARO requires positive rank_weight and tie_cost_weight")
        if pointwise_weight < 0 or caro_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        for name, values in (
            ("embeddings", embeddings),
            ("performance", performance),
            ("cost", costs),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"EquiRouter received non-finite {name} values")

        mse = nn.MSELoss()
        self.cost_mean, self.cost_std = fit_cost_model(
            self.model_cost, embeddings, costs, config, self.device
        )

        performance_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(np.array(embeddings, dtype=np.float32, copy=True)),
                torch.from_numpy(np.array(performance, dtype=np.float32, copy=True)),
                torch.from_numpy(np.array(costs, dtype=np.float32, copy=True)),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        performance_optimizer = make_optimizer(self.model_perf.parameters(), config)
        self.model_perf.train()
        for epoch in range(epochs):
            totals = {"loss": 0.0, "mse": 0.0, "caro": 0.0}
            batches = 0
            for batch_embeddings, batch_performance, batch_costs in performance_loader:
                batch_embeddings = batch_embeddings.to(self.device)
                batch_performance = batch_performance.to(self.device)
                batch_costs = batch_costs.to(self.device)
                performance_optimizer.zero_grad()
                scores = self.model_perf(batch_embeddings)
                mse_loss = mse(scores, batch_performance)
                caro_loss = pairwise_rank_loss(
                    scores,
                    batch_performance,
                    costs=batch_costs,
                    tie_epsilon=tie_epsilon,
                    tie_cost_weight=tie_cost_weight,
                    temperature=temperature,
                    margin=margin,
                )
                loss = pointwise_weight * mse_loss + caro_weight * caro_loss
                loss.backward()
                if gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model_perf.parameters(), gradient_clip)
                performance_optimizer.step()
                totals["loss"] += float(loss.item())
                totals["mse"] += float(mse_loss.item())
                totals["caro"] += float(caro_loss.item())
                batches += 1
            logger.info(
                "[EquiRouter] epoch %d/%d loss=%.6f mse=%.6f caro=%.6f",
                epoch + 1,
                epochs,
                totals["loss"] / max(1, batches),
                totals["mse"] / max(1, batches),
                totals["caro"] / max(1, batches),
            )
        self.model_perf.eval()

    def predict(self, test_embedding, batch_size: int = 512):
        if isinstance(test_embedding, torch.Tensor):
            embeddings = test_embedding.detach().cpu().numpy().astype(np.float32)
        else:
            embeddings = np.asarray(test_embedding, dtype=np.float32)
        expected_dim = int(self.args["embeddings"]["out_dim"])
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            raise ValueError(f"Expected embeddings of shape (N, {expected_dim})")
        if not np.isfinite(embeddings).all() or batch_size <= 0:
            raise ValueError("Embeddings must be finite and batch_size positive")
        if not hasattr(self, "cost_mean"):
            raise RuntimeError("Call train() before predict()")
        if not len(embeddings):
            empty = np.empty((0, len(self.model_list)), dtype=np.float32)
            return empty, empty.copy()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(embeddings)),
            batch_size=batch_size,
            shuffle=False,
        )
        performance_parts: list[torch.Tensor] = []
        cost_parts: list[torch.Tensor] = []
        self.model_cost.eval()
        self.model_perf.eval()
        with torch.no_grad():
            for (batch_embeddings,) in loader:
                batch_embeddings = batch_embeddings.to(self.device)
                standardized_cost = self.model_cost(batch_embeddings)
                predicted_cost = (standardized_cost * self.cost_std + self.cost_mean).clamp_min(0.0)
                performance_parts.append(self.model_perf(batch_embeddings).cpu())
                cost_parts.append(predicted_cost.cpu())
        return (
            torch.cat(performance_parts).numpy(),
            torch.cat(cost_parts).numpy(),
        )


def pairwise_rank_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    costs: torch.Tensor | None = None,
    tie_epsilon: float = 0.0,
    tie_cost_weight: float = 1.0,
    temperature: float = 1.0,
    margin: float = 0.0,
) -> torch.Tensor:
    """CARO pairwise logistic loss.

    Utility differences larger than ``tie_epsilon`` create strict preference
    pairs. Inside the indifference radius, lower cost determines preference.
    ``tie_epsilon=0`` recovers exact-equality CARO.
    """
    if scores.ndim != 2 or labels.ndim != 2 or scores.shape != labels.shape:
        raise ValueError("scores and labels must have matching shape (B, M)")
    if costs is not None and costs.shape != scores.shape:
        raise ValueError("costs must have the same shape as scores")
    if temperature <= 0 or tie_epsilon < 0 or tie_cost_weight < 0:
        raise ValueError("temperature must be positive and tie settings non-negative")

    with torch.no_grad():
        utility_delta = labels.unsqueeze(2) - labels.unsqueeze(1)
        strict_pairs = utility_delta > tie_epsilon
        cost_pairs = torch.zeros_like(strict_pairs)
        if costs is not None and tie_cost_weight > 0:
            cost_delta = costs.unsqueeze(2) - costs.unsqueeze(1)
            cost_pairs = (utility_delta.abs() <= tie_epsilon) & (cost_delta < 0)
        pairs = strict_pairs | cost_pairs
        weights = torch.ones_like(utility_delta, dtype=scores.dtype)
        weights = torch.where(
            cost_pairs,
            torch.as_tensor(tie_cost_weight, device=scores.device, dtype=scores.dtype),
            weights,
        )

    if not pairs.any():
        return scores.sum() * 0.0
    score_delta = scores.unsqueeze(2) - scores.unsqueeze(1)
    losses = F.softplus((margin - score_delta) / temperature)
    weighted_pairs = weights * pairs.to(scores.dtype)
    numerator = (losses * weighted_pairs).sum(dim=(1, 2))
    denominator = weighted_pairs.sum(dim=(1, 2))
    valid = denominator > 0
    return (numerator[valid] / denominator[valid]).mean()
