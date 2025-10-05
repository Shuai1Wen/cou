# -*- coding: utf-8 -*-
"""Stability utilities for linear generators."""

from __future__ import annotations

import numpy as np


def sym_part_max_eig(L: np.ndarray, n_iter: int = 50) -> float:
    if L.size == 0:
        return 0.0
    S = 0.5 * (L + L.T)
    rng = np.random.default_rng(0)
    v = rng.normal(size=S.shape[1])
    v /= np.linalg.norm(v) + 1e-8
    for _ in range(n_iter):
        v = S @ v
        norm = np.linalg.norm(v) + 1e-8
        v /= norm
    return float(v @ (S @ v))


def spectral_abscissa(L: np.ndarray) -> float:
    if L.size == 0:
        return 0.0
    eigvals = np.linalg.eigvals(L)
    if eigvals.size == 0:
        return 0.0
    return float(np.max(np.real(eigvals)))


def mu2_log_norm(L: np.ndarray) -> float:
    if L.size == 0:
        return 0.0
    S = 0.5 * (L + L.T)
    eigvals = np.linalg.eigvalsh(S)
    if eigvals.size == 0:
        return 0.0
    return float(np.max(eigvals))


def project_to_stable(
    L: np.ndarray,
    *,
    margin: float = -1e-3,
    enable: bool = True,
    soft_weight: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project generator to satisfy logarithmic norm margin.

    Returns the projected matrix together with diagnostics containing the
    spectral abscissa/logarithmic norm before and after projection.
    """

    if L.size == 0:
        diagnostics = {
            "alpha_raw": 0.0,
            "mu2_raw": 0.0,
            "alpha_proj": 0.0,
            "mu2_proj": 0.0,
            "violation": 0.0,
            "margin": margin,
        }
        return L, diagnostics

    alpha_raw = spectral_abscissa(L)
    mu2_raw = mu2_log_norm(L)

    if not enable:
        diagnostics = {
            "alpha_raw": alpha_raw,
            "mu2_raw": mu2_raw,
            "alpha_proj": alpha_raw,
            "mu2_proj": mu2_raw,
            "violation": max(0.0, mu2_raw - margin),
            "margin": margin,
        }
        return L, diagnostics

    S = 0.5 * (L + L.T)
    w, Q = np.linalg.eigh(S)
    w_clipped = np.minimum(w, margin)
    S_proj = (Q * w_clipped) @ Q.T
    A = 0.5 * (L - L.T)
    L_hard = S_proj + A

    soft_weight = float(np.clip(soft_weight, 0.0, 1.0))
    if soft_weight < 1.0:
        L_proj = L + soft_weight * (L_hard - L)
        mu2_soft = mu2_log_norm(L_proj)
        if mu2_soft > margin + 1e-9:
            L_proj = L_hard
    else:
        L_proj = L_hard

    alpha_proj = spectral_abscissa(L_proj)
    mu2_proj = mu2_log_norm(L_proj)

    diagnostics = {
        "alpha_raw": alpha_raw,
        "mu2_raw": mu2_raw,
        "alpha_proj": alpha_proj,
        "mu2_proj": mu2_proj,
        "violation": max(0.0, mu2_raw - margin),
        "margin": margin,
        "soft_weight": soft_weight,
    }
    return L_proj, diagnostics


def soft_stability_penalty(
    L: np.ndarray,
    alpha: float = 1e-3,
    lambda_stab: float = 1.0,
) -> float:
    """Compute soft stability penalty: lambda * max(0, mu2 + alpha)^2.

    Args:
        L: Generator matrix
        alpha: Stability margin (negative, e.g., -1e-3)
        lambda_stab: Penalty weight

    Returns:
        penalty: Soft penalty value
    """
    if L.size == 0:
        return 0.0

    mu2 = mu2_log_norm(L)
    violation = max(0.0, mu2 + abs(alpha))  # mu2 should be <= -alpha
    penalty = lambda_stab * (violation ** 2)
    return float(penalty)


def stability_penalty(values, margin: float = 0.0, lam: float = 1.0) -> float:
    """Legacy stability penalty (linear).

    For quadratic penalty, use soft_stability_penalty instead.
    """
    total = 0.0
    if values is None:
        return 0.0
    for item in values:
        if isinstance(item, np.ndarray) and item.ndim == 2:
            mu2 = mu2_log_norm(item)
        else:
            mu2 = float(item)
        if mu2 > margin:
            total += lam * (mu2 - margin)
    return float(total)


__all__ = [
    "sym_part_max_eig",
    "spectral_abscissa",
    "mu2_log_norm",
    "project_to_stable",
    "soft_stability_penalty",
    "stability_penalty",
]
