"""Semigroup generators and stability projection."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, eigh

Array = np.ndarray


def project_stable(L: Array, alpha: float = 1e-3) -> Array:
    """Project generator to stable subspace.

    Projects the symmetric part S=(L+L^T)/2 so that all eigenvalues ≤ -alpha,
    ensuring asymptotic stability of the linear system.

    Parameters
    ----------
    L : Array
        Generator matrix of shape (d, d)
    alpha : float
        Minimum decay rate (positive value)

    Returns
    -------
    L_proj : Array
        Projected stable generator
    """

    # Decompose into symmetric and antisymmetric parts
    S = 0.5 * (L + L.T)  # Symmetric part
    A = 0.5 * (L - L.T)  # Antisymmetric part

    # Eigen-decomposition of symmetric part
    w, Q = eigh(S)

    # Project eigenvalues to ensure stability: λ ≤ -α
    w_proj = np.minimum(w, -alpha * np.ones_like(w))

    # Reconstruct symmetric part
    S_proj = (Q * w_proj) @ Q.T

    # Combine with unchanged antisymmetric part
    return A + S_proj


def pushforward(X: Array, L: Array, tau: float = 0.5) -> Array:
    """Apply semigroup generator for time τ.

    Computes e^{τL} X^T where L is the generator matrix.

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


def spectral_radius(L: Array) -> float:
    """Compute spectral radius of symmetric part.

    Parameters
    ----------
    L : Array
        Generator matrix of shape (d, d)

    Returns
    -------
    radius : float
        Maximum real eigenvalue of (L + L^T)/2
    """
    S = 0.5 * (L + L.T)
    eigenvals = np.linalg.eigvals(S)
    max_real = float(np.max(np.real(eigenvals)))
    if max_real > 0:
        return max_real * 0.5
    return max_real


__all__ = ["project_stable", "pushforward", "spectral_radius"]