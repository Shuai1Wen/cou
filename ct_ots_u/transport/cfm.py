"""Conditional flow matching (CFM-Lite) vector field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.nn.utils import spectral_norm
except ImportError as exc:  # pragma: no cover - safety guard
    raise ImportError("ct_ots_u.transport.cfm requires PyTorch.") from exc

__all__ = ["CFMConfig", "CondFlowField"]


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
class CFMConfig:
    """Configuration for :class:`CondFlowField`."""

    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.1
    activation: Activation = "gelu"
    layer_norm: bool = True
    spectral_norm: bool = True
    time_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)


class CondFlowField(nn.Module):
    """Conditional vector field trained via flow matching regression."""

    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        *,
        config: Optional[CFMConfig] = None,
    ) -> None:
        super().__init__()
        cfg = config or CFMConfig()
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
        input_dim = self.embed_dim + self.cond_dim + 1  # +1 for time
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

        if len(cfg.time_grid) == 0:
            raise ValueError("time_grid must contain at least one entry")
        self.register_buffer(
            "time_grid",
            torch.tensor(cfg.time_grid, dtype=torch.float32),
        )

    def forward(self, h: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        if h.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected h dim {self.embed_dim}, got {h.shape[-1]}")
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.shape[-1] != 1:
            raise ValueError("Time input must broadcast to a single scalar per sample")
        x = torch.cat([h, t, cond], dim=-1)
        return self.net(x)

    def predict(self, h0: Tensor, cond: Tensor, *, steps: int = 8) -> Tensor:
        """Integrate the learned ODE using simple Euler steps."""

        if steps <= 0:
            raise ValueError("steps must be positive")
        dt = 1.0 / float(steps)
        h = h0
        for k in range(steps):
            t = h0.new_full((h0.shape[0], 1), (k + 0.5) * dt)
            v = self.forward(h, t, cond)
            h = h + dt * v
        return h

    def loss(self, pred: Tensor, target: Tensor, *, kind: str = "huber", delta: float = 1.0) -> Tensor:
        if kind == "huber":
            return F.huber_loss(pred, target, delta=float(delta))
        if kind == "mse":
            return F.mse_loss(pred, target)
        raise ValueError(f"Unsupported loss kind '{kind}'")

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample discrete times from the configured grid (teacher forcing)."""

        if self.time_grid.numel() == 1:
            return self.time_grid[0].expand(batch_size, 1).to(device=device)
        idx = torch.randint(0, self.time_grid.numel(), (batch_size,), device=device)
        t = self.time_grid.index_select(0, idx)
        return t.unsqueeze(1)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        grid = ",".join(f"{float(t):.2f}" for t in self.time_grid.tolist())
        return (
            f"embed_dim={self.embed_dim}, cond_dim={self.cond_dim}, "
            f"hidden={self.config.hidden_dim}, layers={self.config.layers}, "
            f"time_grid=[{grid}]"
        )
