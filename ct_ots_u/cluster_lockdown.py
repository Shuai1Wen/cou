"""Cluster stabilization helpers for self-R experiments."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from .gating_kselect import enforce_min_cluster, merge_components_wasserstein, train_gating
from .ct_ots_model import GatedSemigroup
from .ot_metrics import sinkhorn_divergence_uot


def icl_bic_stats(
    X: np.ndarray,
    covariance_type: str,
    k_max: int,
    seed: int,
) -> List[dict]:
    results: List[dict] = []
    for k in range(1, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=1e-4,
            random_state=seed,
        ).fit(X)
        bic = float(gmm.bic(X))
        resp = gmm.predict_proba(X)
        entropy = -2.0 * float(np.sum(resp * np.log(resp + 1e-12)))
        icl = bic + entropy
        results.append(
            {
                "K": int(k),
                "bic": bic,
                "icl": float(icl),
                "entropy": -0.5 * entropy,
            }
        )
    return results


def cluster_sensitivity(
    Xt2: np.ndarray,
    X0: np.ndarray,
    Xt1: np.ndarray,
    Xt12: np.ndarray,
    gmm: GaussianMixture,
    rank: int,
    reg: float,
    reg_m: float,
    tau: float,
    top_p: float,
    min_branch: int,
    min_frac_grid: Iterable[float],
    wass_grid: Iterable[float],
    num_iter_max: int = 1000,
    stop_thr: float = 1e-6,
) -> List[dict]:
    records: List[dict] = []
    for min_frac in min_frac_grid:
        labels_enforced, actual_frac = enforce_min_cluster(gmm, Xt2, min_frac=min_frac)
        for thresh in wass_grid:
            labels_merged, K_final = merge_components_wasserstein(Xt2, labels_enforced, threshold=thresh)
            responsibilities = _one_hot(labels_merged, K_final)
            gating_temp = train_gating(Xt2, responsibilities, max_iter=300)
            model = GatedSemigroup(
                K=K_final,
                rank=rank,
                reg=reg,
                reg_m=reg_m,
                stable_margin=-1e-3,
                log_diagnostics=False,
                stable_alpha=1e-3,
                min_branch_samples=min_branch,
                tau=tau,
                num_iter_max=num_iter_max,
                stop_thr=stop_thr,
                train_steps=150,
                train_lr=1e-2,
            ).fit_branchwise(X0, Xt1, gating_model=gating_temp, top_p=top_p)
            pred = model.forward_steps(X0, steps=2, delta=1.0)
            err = sinkhorn_divergence_uot(
                pred,
                Xt12,
                reg=reg,
                reg_m=reg_m,
                num_iter_max=num_iter_max,
                stop_thr=stop_thr,
            )
            records.append(
                {
                    "min_cluster_frac": float(min_frac),
                    "merge_wass_thresh": float(thresh),
                    "K_final": int(K_final),
                    "min_cluster_actual": float(actual_frac),
                    "uot_error": float(err),
                    "stability_penalty": float(model.stability_penalty),
                }
            )
    return records


def _one_hot(labels: np.ndarray, k: int) -> np.ndarray:
    mat = np.zeros((labels.size, k), dtype=float)
    mat[np.arange(labels.size), labels] = 1.0
    return mat
