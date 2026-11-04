"""Neural cost model used by EquiRouter."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
    }
    try:
        return activations[name.casefold()]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


class MLP(nn.Module):
    """Configurable MLP with the optional bottleneck used by RouterEval."""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        use_mf: bool = False,
        emb_dim: int | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if use_mf and emb_dim is None:
            raise ValueError("emb_dim is required when use_mf is true")
        self.use_mf = bool(use_mf)
        self.mf_encoder = nn.Linear(input_dim, emb_dim) if use_mf else None
        self.mf_decoder = nn.Linear(emb_dim, input_dim) if use_mf else None

        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_sizes:
            width = int(width)
            layers.append(nn.Linear(previous, width))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        self.mlp_body = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_layer = nn.Linear(previous, out_dim)
        self._init_weights(float(init_std))

    def _init_weights(self, standard_deviation: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=standard_deviation)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_mf:
            inputs = self.mf_decoder(self.mf_encoder(inputs))
        return self.output_layer(self.mlp_body(inputs))


def init_model(args: dict, input_dim: int, out_dim: int) -> MLP:
    """Build the single cost-model architecture supported by EquiRouter."""
    config = args["model"]
    if str(config.get("name", "")).casefold() != "mlp":
        raise ValueError("EquiRouter supports only the MLP cost model")
    matrix_factorization = config.get("mf", {})
    return MLP(
        input_dim=input_dim,
        out_dim=out_dim,
        hidden_sizes=config.get("hidden_sizes", ()),
        activation=config.get("activation", "relu"),
        dropout=float(config.get("dropout", 0.0)),
        use_batchnorm=bool(config.get("use_batchnorm", False)),
        use_mf=bool(matrix_factorization.get("use_mf", False)),
        emb_dim=input_dim // 4,
        init_std=float(matrix_factorization.get("init_std", 0.02)),
    )
