"""Wasserstein-based cluster merging."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import sqrtm

Array = np.ndarray


def w2_gaussian(mu1: Array, C1: Array, mu2: Array, C2: Array) -> float:
    """Compute 2-Wasserstein distance between two Gaussian distributions.

    Uses the closed-form formula for Gaussian distributions:
    W_2^2(N(μ₁,C₁), N(μ₂,C₂)) = ||μ₁-μ₂||² + tr(C₁ + C₂ - 2√(C₁^(1/2) C₂ C₁^(1/2)))

    Parameters
    ----------
    mu1, mu2 : Array
        Mean vectors of shape (d,)
    C1, C2 : Array
        Covariance matrices of shape (d, d)

    Returns
    -------
    distance : float
        2-Wasserstein distance
    """

    # Mean difference term
    mean_diff = float(np.linalg.norm(mu1 - mu2)**2)

    # Covariance term using matrix square root
    sqrt_C1 = sqrtm(C1)
    cross_term = sqrtm(sqrt_C1 @ C2 @ sqrt_C1)

    # Handle complex numbers from numerical errors
    if np.iscomplexobj(cross_term):
        cross_term = cross_term.real

    cov_term = float(np.trace(C1 + C2 - 2 * cross_term))

    return mean_diff + cov_term


def merge_by_w2(X: Array, labels: Array, w_thresh: float = 0.5) -> Tuple[Array, int]:
    """Merge clusters based on Wasserstein-2 distance threshold.

    Parameters
    ----------
    X : Array
        Data matrix of shape (n, d)
    labels : Array
        Cluster labels of shape (n,)
    w_thresh : float
        Wasserstein distance threshold for merging

    Returns
    -------
    new_labels : Array
        Updated cluster labels after merging
    n_clusters : int
        Number of clusters after merging
    """

    K = int(labels.max()) + 1
    d = X.shape[1]

    # Estimate Gaussian parameters for each cluster
    mus = []
    covs = []

    for k in range(K):
        Xk = X[labels == k]
        if len(Xk) == 0:
            # Empty cluster
            mus.append(np.zeros(d))
            covs.append(np.eye(d) * 1e-5)
        elif len(Xk) == 1:
            mus.append(Xk[0])
            covs.append(np.eye(d) * 1e-5)
        else:
            mus.append(Xk.mean(axis=0))
            cov_k = np.cov(Xk.T, bias=False) + 1e-5 * np.eye(d)  # Regularized covariance
            covs.append(cov_k)

    # Union-find data structure for merging
    parent = list(range(K))

    def find(x: int) -> int:
        """Find root with path compression."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        """Union two components."""
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Check all pairs for merging
    for i in range(K):
        for j in range(i + 1, K):
            distance = w2_gaussian(mus[i], covs[i], mus[j], covs[j])
            if distance < w_thresh:
                union(i, j)

    # Create new label mapping
    roots = {}
    new_label = 0
    for i in range(K):
        root = find(i)
        if root not in roots:
            roots[root] = new_label
            new_label += 1

    # Apply new labels
    new_labels = np.array([roots[find(label)] for label in labels])

    return new_labels, len(roots)


__all__ = ["w2_gaussian", "merge_by_w2"]