"""Conditional residual regressor (CRR) transport model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.nn.utils import spectral_norm
except ImportError as exc:  # pragma: no cover - safety guard
    raise ImportError("ct_ots_u.transport.crr requires PyTorch.") from exc

__all__ = ["CRRConfig", "CondResidualRegressor"]


Activation = Literal["gelu", "relu", "silu"]


def _make_activation(name: Activation) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}'")


@dataclass
class CRRConfig:
    """Hyper-parameters for :class:`CondResidualRegressor`."""

    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.1
    activation: Activation = "gelu"
    layer_norm: bool = True
    spectral_norm: bool = True


class CondResidualRegressor(nn.Module):
    """Minimal conditional residual network for distribution matching.

    The module receives a base embedding ``h0`` together with a conditioning
    vector ``cond`` (perturbation identifiers, covariates, etc.) and predicts a
    residual correction that approximates the perturbation effect. The final
    perturbed embedding is obtained via ``h0 + residual``.
    """

    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        *,
        config: Optional[CRRConfig] = None,
    ) -> None:
        super().__init__()
        cfg = config or CRRConfig()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if cond_dim <= 0:
            raise ValueError("cond_dim must be positive")
        if cfg.layers < 1:
            raise ValueError("layers must be ≥ 1")
        self.embed_dim = int(embed_dim)
        self.cond_dim = int(cond_dim)
        self.config = cfg

        layers: list[nn.Module] = []
        input_dim = self.embed_dim + self.cond_dim
        hidden = int(cfg.hidden_dim)
        activation = _make_activation(cfg.activation)

        for layer_idx in range(cfg.layers):
            out_dim = hidden if layer_idx < cfg.layers - 1 else self.embed_dim
            linear = nn.Linear(input_dim, out_dim)
            if cfg.spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)
            if layer_idx < cfg.layers - 1:
                if cfg.layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                layers.append(activation)
                if cfg.dropout > 0.0:
                    layers.append(nn.Dropout(cfg.dropout))
                input_dim = hidden

        self.net = nn.Sequential(*layers)

    def forward(self, h0: Tensor, cond: Tensor) -> Tensor:
        """Return residual vector conditioned on ``cond``."""

        if h0.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected h0 dim {self.embed_dim}, got {h0.shape[-1]}")
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")
        x = torch.cat([h0, cond], dim=-1)
        return self.net(x)

    def predict(self, h0: Tensor, cond: Tensor) -> Tensor:
        """Convenience wrapper returning the perturbed embedding."""

        return h0 + self.forward(h0, cond)

    def loss(self, h_hat: Tensor, target: Tensor, *, kind: str = "huber", delta: float = 1.0) -> Tensor:
        """Compute regression loss used during training."""

        if kind == "huber":
            return F.huber_loss(h_hat, target, delta=float(delta))
        if kind == "mse":
            return F.mse_loss(h_hat, target)
        raise ValueError(f"Unsupported loss kind '{kind}'")

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"embed_dim={self.embed_dim}, cond_dim={self.cond_dim}, "
            f"hidden={self.config.hidden_dim}, layers={self.config.layers}"
        )
