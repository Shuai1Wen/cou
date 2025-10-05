from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["StableLinear", "StructuredParams"]


@dataclass
class StructuredParams:
    """Container for structured linear parameters."""

    U: torch.Tensor
    V: torch.Tensor
    W: torch.Tensor
    log_gamma: torch.Tensor

    def mean_reduce(self) -> "StructuredParams":
        """Reduce parameters over the leading batch dimension via mean."""
        if self.U.dim() > 2:
            U = self.U.mean(dim=0)
            V = self.V.mean(dim=0)
            W = self.W.mean(dim=0)
        else:
            U, V, W = self.U, self.V, self.W
        log_gamma = self.log_gamma.mean() if self.log_gamma.dim() > 0 else self.log_gamma
        return StructuredParams(U=U, V=V, W=W, log_gamma=log_gamma)

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.U, self.V, self.W, self.log_gamma


class StableLinear(nn.Module):
    """Structured linear operator with built-in stability bias.

    Implements
        L = skew(U V^T) - W W^T - gamma * I
    where gamma = exp(log_gamma) >= 0.
    """

    def __init__(
        self,
        d: int,
        r: Optional[int] = None,
        *,
        gamma_init: float = 1e-3,
        learn_gamma: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        r = r or min(32, d)
        generator = torch.Generator(device=device).manual_seed(torch.randint(0, 10_000, (1,)).item())
        self.U = nn.Parameter(torch.randn(d, r, generator=generator, device=device, dtype=dtype) * 1e-2)
        self.V = nn.Parameter(torch.randn(d, r, generator=generator, device=device, dtype=dtype) * 1e-2)
        self.W = nn.Parameter(torch.randn(d, r, generator=generator, device=device, dtype=dtype) * 1e-2)
        if learn_gamma:
            init = torch.tensor(gamma_init, device=device, dtype=dtype).clamp(min=1e-8)
            self.log_gamma = nn.Parameter(torch.log(init))
        else:
            self.register_buffer("log_gamma", torch.tensor(gamma_init, device=device, dtype=dtype))
        self.register_buffer("eye", torch.eye(d, device=device, dtype=dtype))

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    def forward(
        self,
        overrides: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if overrides is not None:
            U, V, W, log_gamma = overrides
        else:
            U, V, W, log_gamma = self.U, self.V, self.W, self.log_gamma
        K = U @ V.T
        skew = 0.5 * (K - K.T)
        diss = W @ W.T
        gamma = log_gamma.exp() if overrides is not None else self.gamma
        return skew - diss - gamma * self.eye

    def params_tuple(self) -> dict[str, torch.Tensor | float]:
        return {
            "U": self.U,
            "V": self.V,
            "W": self.W,
            "gamma": float(self.gamma.detach().cpu()),
        }
