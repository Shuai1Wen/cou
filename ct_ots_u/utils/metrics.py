"""Metrics and distance computation utilities."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

Array = np.ndarray


def energy_distance(X: Array, Y: Array) -> float:
    """Compute energy distance between two point clouds.

    The energy distance is defined as:
    E(X,Y) = 2⟨d(X,Y)⟩ - ⟨d(X,X)⟩ - ⟨d(Y,Y)⟩

    where ⟨d(A,B)⟩ is the expected distance between points from A and B.

    Parameters
    ----------
    X : Array
        First point cloud of shape (n, d)
    Y : Array
        Second point cloud of shape (m, d)

    Returns
    -------
    energy : float
        Energy distance between X and Y
    """

    def mean_distance(A: Array, B: Array) -> float:
        """Mean pairwise distance between two point sets."""
        if A.shape[0] == 0 or B.shape[0] == 0:
            return 0.0
        dists = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        return float(dists.mean())

    # Cross-distances
    d_xy = mean_distance(X, Y)

    # Within-distances
    d_xx = mean_distance(X, X)
    d_yy = mean_distance(Y, Y)

    return 2 * d_xy - d_xx - d_yy


def maximum_mean_discrepancy(
    X: Array,
    Y: Array,
    kernel: Literal["rbf", "linear"] = "rbf",
    gamma: float = 1.0
) -> float:
    """Compute Maximum Mean Discrepancy between two distributions.

    Parameters
    ----------
    X : Array
        First sample of shape (n, d)
    Y : Array
        Second sample of shape (m, d)
    kernel : str
        Kernel type ("rbf" or "linear")
    gamma : float
        Kernel bandwidth parameter (for RBF kernel)

    Returns
    -------
    mmd : float
        MMD estimate
    """

    def kernel_matrix(A: Array, B: Array) -> Array:
        """Compute kernel matrix between A and B."""
        if kernel == "rbf":
            # RBF kernel: k(x,y) = exp(-γ||x-y||²)
            dists_sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
            return np.exp(-gamma * dists_sq)
        elif kernel == "linear":
            # Linear kernel: k(x,y) = x^T y
            return A @ B.T
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    # Kernel matrices
    K_xx = kernel_matrix(X, X)
    K_yy = kernel_matrix(Y, Y)
    K_xy = kernel_matrix(X, Y)

    # MMD² = ⟨φ(X),φ(X)⟩ + ⟨φ(Y),φ(Y)⟩ - 2⟨φ(X),φ(Y)⟩
    n, m = X.shape[0], Y.shape[0]
    mmd_sq = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1))
    mmd_sq += (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1))
    mmd_sq -= 2 * K_xy.mean()

    return float(np.sqrt(max(0.0, mmd_sq)))


def compute_cost_matrix(
    X: Array,
    Y: Array,
    metric: Literal["euclidean", "sqeuclidean", "cosine"] = "euclidean"
) -> Array:
    """Compute cost matrix between two point clouds.

    Parameters
    ----------
    X : Array
        First point cloud of shape (n, d)
    Y : Array
        Second point cloud of shape (m, d)
    metric : str
        Distance metric

    Returns
    -------
    M : Array
        Cost matrix of shape (n, m)
    """

    if metric == "euclidean":
        return np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    elif metric == "sqeuclidean":
        return np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    elif metric == "cosine":
        # Cosine distance: 1 - cos(x,y) = 1 - x^T y / (||x|| ||y||)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        cos_sim = X_norm @ Y_norm.T
        return 1.0 - cos_sim
    else:
        raise ValueError(f"Unknown metric: {metric}")


def wasserstein_barycenter_weights(
    X: Array,
    Y: Array,
    transport_matrix: Array
) -> Array:
    """Compute barycentric mapping weights from transport matrix.

    Given optimal transport matrix G, computes the barycentric mapping
    T(x) = Σⱼ G(i,j)/a(i) * yⱼ where a(i) = Σⱼ G(i,j).

    Parameters
    ----------
    X : Array
        Source points of shape (n, d)
    Y : Array
        Target points of shape (m, d)
    transport_matrix : Array
        Transport matrix G of shape (n, m)

    Returns
    -------
    T : Array
        Mapped points of shape (n, d)
    """

    # Normalize transport matrix row-wise
    row_sums = transport_matrix.sum(axis=1, keepdims=True)
    weights = transport_matrix / (row_sums + 1e-8)

    # Compute barycentric mapping
    T = weights @ Y

    return T


def alignment_score(X: Array, Y: Array) -> float:
    """Compute alignment score between two point clouds.

    Uses Procrustes analysis to find optimal alignment and returns
    the normalized Frobenius distance.

    Parameters
    ----------
    X : Array
        First point cloud of shape (n, d)
    Y : Array
        Second point cloud of shape (n, d) - must have same n as X

    Returns
    -------
    score : float
        Alignment score (0 = perfect alignment, higher = worse)
    """

    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: X{X.shape} vs Y{Y.shape}")

    # Center the data
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    # Compute cross-covariance matrix
    H = X_centered.T @ Y_centered

    # SVD for optimal rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply optimal transformation
    Y_aligned = Y_centered @ R.T

    # Compute alignment error
    error = np.linalg.norm(X_centered - Y_aligned, 'fro')
    scale = max(np.linalg.norm(X_centered, 'fro'), np.linalg.norm(Y_centered, 'fro'))

    return float(error / (scale + 1e-8))


__all__ = [
    "energy_distance",
    "maximum_mean_discrepancy",
    "compute_cost_matrix",
    "wasserstein_barycenter_weights",
    "alignment_score"
]