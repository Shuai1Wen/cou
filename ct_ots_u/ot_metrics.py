# -*- coding: utf-8 -*-
"""Optimal transport metric helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .ot.uot_losses import sinkhorn_divergence, uot_sinkhorn_cost

Array = np.ndarray


def sinkhorn_divergence_bal(
    X: Array,
    Y: Array,
    reg: float = 0.05,
    *,
    backend: str = "geomloss",
    sinkhorn_backend: str = "online",
    scaling: float = 0.9,
    num_iter_max: int = 1000,
    stop_thr: float = 1e-6,
) -> float:
    return float(
        sinkhorn_divergence(
            X,
            Y,
            blur=reg,
            backend=backend,
            sinkhorn_backend=sinkhorn_backend,
            scaling=scaling,
            num_iter_max=num_iter_max,
            stop_thr=stop_thr,
        )
    )


def sinkhorn_divergence_uot(
    X: Array,
    Y: Array,
    reg: float = 0.05,
    reg_m: float = 1.0,
    *,
    backend: str = "geomloss",
    sinkhorn_backend: str = "online",
    scaling: float = 0.9,
    num_iter_max: int = 1000,
    stop_thr: float = 1e-6,
) -> float:
    cost_xy, _, _ = uot_sinkhorn_cost(
        X,
        Y,
        reg=reg,
        reg_m=reg_m,
        metric="sqeuclidean",
        num_iter_max=num_iter_max,
        stop_thr=stop_thr,
        backend=backend,
        sinkhorn_backend=sinkhorn_backend,
        scaling=scaling,
    )
    cost_xx, _, _ = uot_sinkhorn_cost(
        X,
        X,
        reg=reg,
        reg_m=reg_m,
        metric="sqeuclidean",
        num_iter_max=num_iter_max,
        stop_thr=stop_thr,
        backend=backend,
        sinkhorn_backend=sinkhorn_backend,
        scaling=scaling,
    )
    cost_yy, _, _ = uot_sinkhorn_cost(
        Y,
        Y,
        reg=reg,
        reg_m=reg_m,
        metric="sqeuclidean",
        num_iter_max=num_iter_max,
        stop_thr=stop_thr,
        backend=backend,
        sinkhorn_backend=sinkhorn_backend,
        scaling=scaling,
    )
    return float(cost_xy - 0.5 * (cost_xx + cost_yy))


def energy_distance(X: Array, Y: Array) -> float:
    XX = cdist(X, X)
    YY = cdist(Y, Y)
    XY = cdist(X, Y)
    exy = 2.0 * XY.mean()
    exx = XX[np.triu_indices_from(XX, k=1)].mean() * 2.0
    eyy = YY[np.triu_indices_from(YY, k=1)].mean() * 2.0
    return float(exy - exx - eyy)


def compute_uot_distance(
    X: Array,
    Y: Array,
    eps: float = 0.08,
    reg_m: float = 1.0,
    **kwargs
) -> Tuple[float, Array, Array]:
    """Compute UOT distance using POT sinkhorn_unbalanced.

    Args:
        X: Source samples [n, d]
        Y: Target samples [m, d]
        eps: Entropic regularization
        reg_m: Marginal relaxation (unbalanced penalty)
        **kwargs: Additional arguments

    Returns:
        (cost, plan, M): UOT cost, transport plan, cost matrix
    """
    cost, plan, M = uot_sinkhorn_cost(
        X, Y,
        reg=eps,
        reg_m=reg_m,
        metric="sqeuclidean",
        **kwargs
    )
    return float(cost), plan, M


__all__ = [
    "sinkhorn_divergence_bal",
    "sinkhorn_divergence_uot",
    "energy_distance",
    "compute_uot_distance",
]
