# -*- coding: utf-8 -*-
"""Hyperparameter plateau detection for CT-OTS-U."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

Array = np.ndarray


def tau_eps_grid(
    Xs: Array,
    Xt: Array,
    taus: Iterable[float] = (0.5, 1.0, 2.0),
    epsilons: Iterable[float] = (0.03, 0.05, 0.08),
    reg_ms: Iterable[float] = (1.0,),
    repeats: int = 3,
) -> Dict:
    """Scan (τ, ε, reg_m) hyperparameter grid for optimal transport parameters."""

    from ..ot import uot_sinkhorn_cost

    rec = []
    for tau in taus:
        for eps in epsilons:
            for reg_m in reg_ms:
                vals = []
                for _ in range(repeats):
                    error, _, _ = uot_sinkhorn_cost(Xs, Xt, reg=eps, reg_m=float(reg_m))
                    vals.append(error)

                vals_array = np.array(vals)
                mean_val = float(vals_array.mean())
                se_val = (
                    float(vals_array.std(ddof=1) / np.sqrt(len(vals_array))) if len(vals_array) > 1 else 0.0
                )

                rec.append((float(tau), float(eps), float(reg_m), mean_val, se_val))

    best = min(rec, key=lambda x: x[3])  # Minimize mean error
    one_se_threshold = best[3] + best[4]

    plateau = [param for param in rec if param[3] <= one_se_threshold]

    return {
        'best': best,
        'one_se': one_se_threshold,
        'plateau': plateau,
        'grid': rec,
    }


def cross_domain_plateau_check(
    source_plateau: List[Tuple],
    X_target_src: Array,
    X_target_dst: Array,
    eps_grid: Iterable[float] = (0.03, 0.05, 0.08),
    reg_m_grid: Iterable[float] = (0.5, 1.0, 2.0),
    n_repeats: int = 3,
    se_tolerance: float = 1.5,
) -> Dict:
    """Check if target domain UOT falls within source domain plateau.

    Args:
        source_plateau: List of (tau, eps, reg_m, mean, se) from source domain
        X_target_src: Target domain source samples
        X_target_dst: Target domain destination samples
        eps_grid: Epsilon grid for target domain search
        reg_m_grid: reg_m grid for target domain search
        n_repeats: Number of bootstrap repeats
        se_tolerance: Multiple of source SE for plateau inclusion

    Returns:
        decision_dict with:
            - in_plateau: bool
            - source_best: Source domain best point
            - target_best: Target domain best point
            - nearest_in_plateau: Nearest plateau point to target
            - delta_uot: Difference between source and target
    """
    from ..ot import uot_sinkhorn_cost

    # Extract source best point
    source_best = min(source_plateau, key=lambda x: x[3])
    source_mean, source_se = source_best[3], source_best[4]

    # Compute target domain UOT grid
    target_grid = []
    for eps in eps_grid:
        for reg_m in reg_m_grid:
            vals = []
            for _ in range(n_repeats):
                error, _, _ = uot_sinkhorn_cost(
                    X_target_src,
                    X_target_dst,
                    reg=eps,
                    reg_m=float(reg_m),
                    backend='pot',
                )
                vals.append(error)

            vals_array = np.array(vals)
            mean_val = float(vals_array.mean())
            se_val = float(vals_array.std(ddof=1) / np.sqrt(len(vals_array)))

            target_grid.append((0.5, float(eps), float(reg_m), mean_val, se_val))

    target_best = min(target_grid, key=lambda x: x[3])

    # Check if target is in source plateau
    delta_uot = abs(target_best[3] - source_mean)
    in_plateau = delta_uot <= (se_tolerance * source_se)

    # Find nearest plateau point
    if not in_plateau:
        # Find point in source plateau closest to target best
        nearest = min(
            source_plateau,
            key=lambda p: abs(p[3] - target_best[3])
        )
    else:
        nearest = source_best

    return {
        'in_plateau': bool(in_plateau),
        'source_best': {
            'tau': source_best[0],
            'eps': source_best[1],
            'reg_m': source_best[2],
            'mean': source_best[3],
            'se': source_best[4],
        },
        'target_best': {
            'tau': target_best[0],
            'eps': target_best[1],
            'reg_m': target_best[2],
            'mean': target_best[3],
            'se': target_best[4],
        },
        'nearest_in_plateau': {
            'tau': nearest[0],
            'eps': nearest[1],
            'reg_m': nearest[2],
            'mean': nearest[3],
            'se': nearest[4],
        },
        'delta_uot': float(delta_uot),
        'se_threshold': float(se_tolerance * source_se),
    }


__all__ = ["tau_eps_grid", "cross_domain_plateau_check"]
