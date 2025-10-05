"""GMM-based branch selection and gating helpers (legacy wrappers)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from .gating.gmm_kselect import fit_gmm_BICICL, _fit_gaussian_mixture, get_active_components
from .gating.wmerge import w2_gaussian


def _cluster_statistics(X: np.ndarray, labels: np.ndarray, eps: float = 1e-5):
    unique = np.unique(labels)
    means = {}
    covs = {}
    d = X.shape[1]
    for k in unique:
        Xk = X[labels == k]
        if Xk.size == 0:
            means[k] = np.zeros(d)
            covs[k] = np.eye(d) * eps
        else:
            means[k] = Xk.mean(axis=0)
            covs[k] = np.cov(Xk.T) + eps * np.eye(d)
    return means, covs


def gmm_kselect(
    X: np.ndarray,
    K_min: int = 1,
    K_max: int = 10,
    *,
    mode: str = 'bayesian',
    covariance_type: str = 'full',
    random_state: int = 0,
    reg_covar: float = 1e-4,
    weight_concentration_prior: float | None = 1.0,
    gmm_max_iter: int = 500,
    gmm_n_init: int = 5,
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """Legacy wrapper returning BIC/ICL scores for compatibility."""

    k_grid = tuple(range(K_min, K_max + 1))
    best, table = fit_gmm_BICICL(
        X,
        k_grid=k_grid,
        mode=mode,
        random_state=random_state,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        max_iter=gmm_max_iter,
        n_init=gmm_n_init,
        weight_concentration_prior=weight_concentration_prior,
    )

    scores: Dict[int, Dict[str, float]] = {}
    for k, bic, icl, _ in table:
        scores[int(k)] = {"bic": float(bic), "icl": float(icl)}

    best_k = int(best[0]) if best else K_min

    gmm = _fit_gaussian_mixture(
        X,
        best_k,
        mode=mode,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=random_state,
        max_iter=gmm_max_iter,
        n_init=gmm_n_init,
        weight_concentration_prior=weight_concentration_prior,
    )

    resp = gmm.predict_proba(X)
    try:
        bic_retrained = float(gmm.bic(X))
    except AttributeError:
        bic_retrained = float(-2.0 * gmm.score(X) * X.shape[0])
    icl_retrained = bic_retrained - float(np.sum(resp * np.log(resp + 1e-12)))
    scores.setdefault(best_k, {})
    scores[best_k]["bic_retrained"] = float(bic_retrained)
    scores[best_k]["icl_retrained"] = float(icl_retrained)
    return best_k, scores

def enforce_min_cluster(
    gmm: GaussianMixture,
    X: np.ndarray,
    min_frac: float = 0.07,
) -> Tuple[np.ndarray, float]:
    labels = gmm.predict(X)
    n = labels.size
    min_count = max(1, int(np.ceil(min_frac * n)))
    counts = np.bincount(labels, minlength=gmm.n_components)

    while True:
        small = np.where((counts > 0) & (counts < min_count))[0]
        if small.size == 0:
            break
        k = small[0]
        candidates = [j for j in range(gmm.n_components) if j != k and counts[j] > 0]
        if not candidates:
            break
        means, covs = _cluster_statistics(X, labels)
        distances = [w2_gaussian(means[k], covs[k], means[j], covs[j]) for j in candidates]
        target = candidates[int(np.argmin(distances))]
        labels[labels == k] = target
        counts[target] += counts[k]
        counts[k] = 0

    nonzero = counts[counts > 0]
    min_frac_actual = float(nonzero.min() / n) if nonzero.size else 0.0
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    relabeled = np.array([mapping[l] for l in labels], dtype=int)
    return relabeled, min_frac_actual


def merge_components_wasserstein(
    X: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, int]:
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        unique = np.unique(labels)
        if unique.size <= 1:
            break
        means, covs = _cluster_statistics(X, labels)
        best_pair = None
        best_distance = None
        for idx, i in enumerate(unique):
            for j in unique[idx + 1 :]:
                dist = w2_gaussian(means[i], covs[i], means[j], covs[j])
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_pair = (i, j)
        if best_pair is None or best_distance is None or best_distance >= threshold:
            break
        i_min, j_min = best_pair
        labels[labels == j_min] = i_min
        changed = True
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    relabeled = np.array([mapping[l] for l in labels], dtype=int)
    return relabeled, len(unique)


def soft_assign_from_gmm(gmm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    return gmm.predict_proba(X)


def train_gating(
    X: np.ndarray,
    responsibilities: np.ndarray,
    C: float = 1.0,
    max_iter: int = 300,
) -> LogisticRegression:
    labels = responsibilities.argmax(axis=1)
    sample_weight = responsibilities.max(axis=1)
    clf = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
        n_jobs=-1,
    )
    clf.fit(X, labels, sample_weight=sample_weight)
    return clf


__all__ = [
    "gmm_kselect",
    "enforce_min_cluster",
    "merge_components_wasserstein",
    "soft_assign_from_gmm",
    "train_gating",
]
