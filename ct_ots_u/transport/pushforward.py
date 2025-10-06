from __future__ import annotations

from typing import Literal

from torch import Tensor

from ..model.stable_linear import DampedSkewLinear, OrthogonalStep

Mode = Literal["linear"]

__all__ = ["pushforward_torch", "Mode"]


def pushforward_torch(
    X: Tensor,
    dynamics: DampedSkewLinear | OrthogonalStep,
    tau: float,
    *,
    mode: Mode = "linear",
) -> Tensor:
    """Push latent features forward with the configured dynamics."""

    if mode != "linear":
        raise ValueError("Only linear mode is supported in the new dynamics stack")
    return dynamics.step(X, dt=tau)
