"""Semigroup consistency evaluation for CT-OTS-U."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.linalg import expm

Array = np.ndarray


def pushforward(X: Array, L: Array, tau: float = 0.5) -> Array:
    """Apply semigroup generator for time τ.

    Parameters
    ----------
    X : Array
        Input points of shape (n, d)
    L : Array
        Generator matrix of shape (d, d)
    tau : float
        Time step

    Returns
    -------
    Y : Array
        Pushed forward points e^{τL} X^T
    """
    return (expm(tau * L) @ X.T).T


def semigroup_consistency(X0: Array, L1: Array, L2: Array, Xt: Array,
                         tau1: float = 0.5, tau2: float = 0.5,
                         eps: float = 0.08, reg_m: float = 1.0) -> Dict[str, float]:
    """Evaluate semigroup consistency: composed vs direct flow.

    Tests whether the two-step flow X0 → e^{τ₁L₁} X0 → e^{τ₂L₂}(e^{τ₁L₁} X0)
    is consistent with a direct flow X0 → e^{(τ₁+τ₂)L*} X0.

    Parameters
    ----------
    X0 : Array
        Initial points of shape (n, d)
    L1 : Array
        First generator matrix of shape (d, d)
    L2 : Array
        Second generator matrix of shape (d, d)
    Xt : Array
        Target points of shape (m, d)
    tau1, tau2 : float
        Time steps for each stage
    eps : float
        Entropic regularization
    reg_m : float
        Unbalanced regularization

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - err_composed: UOT error for two-step flow
        - err_direct: UOT error for direct flow (using L2 fitted to full interval)
        - improve: Relative improvement (err_direct - err_composed) / err_direct
    """

    # Two-step flow
    X_mid = pushforward(X0, L1, tau=tau1)
    X_composed = pushforward(X_mid, L2, tau=tau2)

    # Direct flow (using L2 for the full time interval)
    X_direct = pushforward(X0, L2, tau=tau1 + tau2)

    # Import UOT cost function
    from ..ot import uot_sinkhorn_cost

    # Compute UOT distances
    err_composed, _, _ = uot_sinkhorn_cost(
        X_composed, Xt, reg=eps, reg_m=reg_m, metric="euclidean"
    )

    err_direct, _, _ = uot_sinkhorn_cost(
        X_direct, Xt, reg=eps, reg_m=reg_m, metric="euclidean"
    )

    # Relative improvement
    improve = (err_direct - err_composed) / max(err_direct, 1e-8)

    return {
        'err_composed': float(err_composed),
        'err_direct': float(err_direct),
        'improve': float(improve)
    }


def stability_margin(L: Array) -> float:
    """Compute spectral margin for stability.

    Parameters
    ----------
    L : Array
        Generator matrix of shape (d, d)

    Returns
    -------
    margin : float
        Maximum real eigenvalue of symmetric part (L + L^T)/2
    """

    S = 0.5 * (L + L.T)  # Symmetric part
    eigenvals = np.linalg.eigvals(S)
    return float(np.max(np.real(eigenvals)))


__all__ = ["semigroup_consistency", "pushforward", "stability_margin"]