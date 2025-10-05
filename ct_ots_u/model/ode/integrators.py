from __future__ import annotations

from typing import Dict, Hashable, Optional, Tuple

import torch
from torch import Tensor

__all__ = ["expm_apply", "step_exp_rnn", "step_split_ode"]


def expm_apply(L: Tensor, X: Tensor, tau: float, *, cache: Optional[Dict[Hashable, Tensor]] = None, key: Hashable | None = None) -> Tensor:
    """Apply matrix exponential to X with optional caching."""

    exp_L = None
    if cache is not None and key is not None:
        exp_L = cache.get(key)
    if exp_L is None:
        exp_L = torch.matrix_exp(L * tau)
        if cache is not None and key is not None:
            cache[key] = exp_L
    return (exp_L @ X.T).T


def step_exp_rnn(
    L: Tensor,
    g,
    X: Tensor,
    tau: float,
    *,
    K: int = 1,
    cache: Optional[Dict[Hashable, Tensor]] = None,
) -> Tensor:
    """Exponentiated RNN integrator."""

    dt = tau / float(max(K, 1))
    cache_key = ("exp_rnn", id(L), dt)
    Y = X
    exp_cache = cache if cache is not None else {}
    for _ in range(max(K, 1)):
        Y_prev = Y
        Y_linear = expm_apply(L, Y_prev, dt, cache=exp_cache, key=cache_key)
        Y = Y_linear + dt * g(Y_prev)
    return Y


def step_split_ode(
    L: Tensor,
    g,
    X: Tensor,
    tau: float,
    *,
    rk: str = "rk2",
    cache: Optional[Dict[Hashable, Tensor]] = None,
) -> Tensor:
    """Strang splitting between linear flow and residual dynamics."""

    half_tau = 0.5 * tau
    cache_key_half = ("split_half", id(L), half_tau)
    exp_cache = cache if cache is not None else {}
    Y0 = expm_apply(L, X, half_tau, cache=exp_cache, key=cache_key_half)
    if rk.lower() == "rk4":
        k1 = g(Y0)
        k2 = g(Y0 + 0.5 * tau * k1)
        k3 = g(Y0 + 0.5 * tau * k2)
        k4 = g(Y0 + tau * k3)
        Ymid = Y0 + (tau / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        k1 = g(Y0)
        k2 = g(Y0 + 0.5 * tau * k1)
        Ymid = Y0 + tau * k2
    Y1 = expm_apply(L, Ymid, half_tau, cache=exp_cache, key=cache_key_half)
    return Y1
