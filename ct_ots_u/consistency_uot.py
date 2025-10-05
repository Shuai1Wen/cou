"""Branch-level UOT consistency utilities."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .ot_metrics import sinkhorn_divergence_uot


def branch_indices_by_top(resp: np.ndarray, k: int, top_p: float = 0.7) -> np.ndarray:
    scores = resp[:, k]
    if top_p <= 0.0 or scores.size == 0:
        return np.arange(scores.size)
    top_p = min(max(top_p, 0.0), 1.0)
    if top_p >= 0.999:
        threshold = np.min(scores) - 1e-8
    else:
        threshold = np.quantile(scores, 1.0 - top_p)
    idx = np.where(scores >= threshold)[0]
    if idx.size == 0:
        idx = np.argsort(scores)[-max(1, int(np.ceil(scores.size * top_p))):]
    return idx


def uot_consistency_branchwise(
    model,
    X0: np.ndarray,
    Xt1: np.ndarray,
    Xt2: np.ndarray,
    Xt12: np.ndarray,
    resp0: np.ndarray,
    resp1: np.ndarray,
    resp2: np.ndarray,
    reg: float,
    reg_m: float,
    top_p: float = 0.7,
    min_samples: int = 50,
    num_iter_max: int = 1000,
    stop_thr: float = 1e-6,
) -> Dict[str, object]:
    K = resp2.shape[1]
    per_branch: List[Dict[str, float]] = []
    total_weight = 0.0
    weighted_cons = 0.0
    weighted_dir = 0.0
    weighted_comp = 0.0
    rel_improvements: List[float] = []

    for k in range(K):
        idx0 = branch_indices_by_top(resp0, k, top_p)
        idx1 = branch_indices_by_top(resp1, k, top_p)
        idx2 = branch_indices_by_top(resp2, k, top_p)
        weight = min(len(idx0), len(idx1), len(idx2))
        if weight < min_samples:
            per_branch.append({"branch": k, "skip": True, "weight": weight})
            continue

        X0k = X0[idx0]
        Xt1k = Xt1[idx1]
        Xt2k = Xt2[idx2]
        Xt12k = Xt12[idx2]

        H1k = model.forward(X0k, delta=1.0)
        H12_comp_k = model.forward(H1k, delta=1.0)
        H12_dir_k = model.forward(X0k, delta=2.0)

        d_comp = sinkhorn_divergence_uot(
            H12_comp_k,
            Xt12k,
            reg=reg,
            reg_m=reg_m,
            num_iter_max=num_iter_max,
            stop_thr=stop_thr,
        )
        d_dir = sinkhorn_divergence_uot(
            H12_dir_k,
            Xt12k,
            reg=reg,
            reg_m=reg_m,
            num_iter_max=num_iter_max,
            stop_thr=stop_thr,
        )
        d_cons = sinkhorn_divergence_uot(
            H12_comp_k,
            H12_dir_k,
            reg=reg,
            reg_m=reg_m,
            num_iter_max=num_iter_max,
            stop_thr=stop_thr,
        )

        total_weight += weight
        weighted_comp += weight * d_comp
        weighted_dir += weight * d_dir
        weighted_cons += weight * d_cons

        rel = (d_dir - d_cons) / max(d_dir, 1e-8)
        rel_improvements.append(rel)

        per_branch.append(
            {
                "branch": k,
                "skip": False,
                "d_comp": float(d_comp),
                "d_dir": float(d_dir),
                "d_cons": float(d_cons),
                "weight": weight,
                "rel_improvement": float(rel),
            }
        )

    aggregate: Dict[str, float] = {}
    if total_weight > 0:
        aggregate = {
            "weighted_d_comp": float(weighted_comp / total_weight),
            "weighted_d_dir": float(weighted_dir / total_weight),
            "weighted_d_cons": float(weighted_cons / total_weight),
            "mean_rel_improvement": float(np.mean(rel_improvements)) if rel_improvements else 0.0,
        }

    return {
        "per_branch": per_branch,
        "aggregate": aggregate,
    }
