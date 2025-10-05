"""Barycentric OT maps with low-rank and stability control."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import ot
from scipy.spatial.distance import cdist

from .stability import project_to_stable


def _barycentric_map(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    reg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.ones(X_src.shape[0]) / X_src.shape[0]
    b = np.ones(X_tgt.shape[0]) / X_tgt.shape[0]
    M = cdist(X_src, X_tgt, metric="sqeuclidean")
    scale = float(M.max())
    if scale > 0:
        M = M / scale
    G = ot.sinkhorn(a, b, M, reg=reg)
    T = (G / (G.sum(axis=1, keepdims=True) + 1e-8)) @ X_tgt
    X_aug = np.hstack([X_src, np.ones((X_src.shape[0], 1))])
    W, *_ = np.linalg.lstsq(X_aug, T - X_src, rcond=None)
    L = W[:-1, :].T
    b_vec = W[-1, :]
    return L, b_vec


def learn_branch_map(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    reg: float = 0.05,
    rank: int | None = None,
    margin: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    L, b_vec = _barycentric_map(X_src, X_tgt, reg=reg)
    if rank is not None and rank < L.shape[0]:
        U, S, Vt = np.linalg.svd(L, full_matrices=False)
        S[rank:] = 0.0
        L = (U * S) @ Vt
    if margin is not None:
        L, _ = project_to_stable(L, margin=margin)
    return L, b_vec

    return L, b_vec
