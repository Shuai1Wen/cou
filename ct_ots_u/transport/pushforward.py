from __future__ import annotations

from typing import Dict, Literal, Optional

from torch import Tensor

from ..model.modules.stable_linear import StableLinear
from ..model.modules.residuals import ResidualNet
from ..model.ode.integrators import expm_apply, step_exp_rnn, step_split_ode

Mode = Literal["linear", "exp_rnn", "ode"]

__all__ = ["pushforward_torch", "Mode"]


def pushforward_torch(
    X: Tensor,
    L_module: StableLinear,
    tau: float,
    *,
    mode: Mode = "linear",
    residual: Optional[ResidualNet] = None,
    K: int = 1,
    rk: str = "rk2",
    cache: Optional[Dict[object, Tensor]] = None,
    L_override: Optional[Tensor] = None,
) -> Tensor:
    """Dispatch pushforward according to selected flow model."""

    if L_override is not None:
        L = L_override
    else:
        L = L_module()
    if mode == "linear":
        return expm_apply(L, X, tau, cache=cache, key=("linear", id(L), tau))
    if residual is None:
        raise ValueError(f"Residual network required for mode '{mode}'")
    if mode == "exp_rnn":
        return step_exp_rnn(L, residual, X, tau, K=K, cache=cache)
    if mode == "ode":
        return step_split_ode(L, residual, X, tau, rk=rk, cache=cache)
    raise ValueError(f"Unknown mode '{mode}'")
