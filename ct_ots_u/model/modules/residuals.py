from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

__all__ = ["SNLinear", "ResidualNet"]


class SNLinear(nn.Linear):
    """Linear layer equipped with spectral normalisation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        power_iterations: int = 1,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        spectral_norm(self, name="weight", n_power_iterations=power_iterations)


class ResidualNet(nn.Module):
    """1-Lipschitz residual block built from spectrally normalised layers."""

    def __init__(
        self,
        d: int,
        *,
        hidden: int = 128,
        depth: int = 2,
        lipschitz_scale: float = 1.0,
        activation: Literal["relu", "silu"] = "relu",
        power_iterations: int = 1,
    ) -> None:
        super().__init__()
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:  # pragma: no cover - guard for invalid input
            raise ValueError(f"Unsupported activation '{activation}'")
        layers: list[nn.Module] = []
        in_features = d
        for _ in range(depth):
            layers.append(SNLinear(in_features, hidden, power_iterations=power_iterations))
            layers.append(act)
            in_features = hidden
        layers.append(SNLinear(in_features, d, power_iterations=power_iterations))
        self.net = nn.Sequential(*layers)
        scale = torch.tensor(float(lipschitz_scale))
        if float(lipschitz_scale) > 1.0 + 1e-6:
            raise ValueError("lipschitz_scale should typically be <= 1.")
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.net(x)
