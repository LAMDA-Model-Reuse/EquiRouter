"""Shared training of the query-conditioned MLP cost predictor."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_optimizer(parameters, config: dict):
    """Construct the optimizer used by both utility and cost predictors."""
    name = str(config.get("optimizer", "adam")).casefold()
    learning_rate = float(config.get("lr", 1e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def fit_cost_model(
    model: nn.Module,
    embeddings: np.ndarray,
    costs: np.ndarray,
    config: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit on training costs and return their inverse-standardization tensors."""
    epochs = int(config.get("epochs", 10))
    batch_size = int(config.get("batch_size", 256))
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if embeddings.ndim != 2 or costs.ndim != 2 or len(embeddings) != len(costs):
        raise ValueError("Training embeddings and costs must be aligned 2D matrices")
    if not len(embeddings) or not costs.shape[1]:
        raise ValueError("Cost training requires queries and candidate models")
    if not np.isfinite(embeddings).all() or not np.isfinite(costs).all():
        raise ValueError("Cost training inputs must be finite")
    if (costs < 0).any():
        raise ValueError("Training costs must be non-negative")

    mean = np.mean(costs, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.std(costs, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    if config.get("standardize_cost", True):
        targets = (costs - mean) / std
    else:
        targets = costs
        mean = np.zeros_like(mean)
        std = np.ones_like(std)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.array(embeddings, dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(targets, dtype=np.float32, copy=True)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = make_optimizer(model.parameters(), config)
    model.train()
    for _ in range(epochs):
        for batch_embeddings, batch_costs in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(
                model(batch_embeddings.to(device)), batch_costs.to(device)
            )
            loss.backward()
            optimizer.step()
    model.eval()
    return torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
