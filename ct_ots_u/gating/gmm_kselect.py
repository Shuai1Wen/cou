# -*- coding: utf-8 -*-
"""GMM/Bayesian gating utilities with BIC/ICL selection and stability helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

Array = np.ndarray


def _fit_gaussian_mixture(
    X: Array,
    k: int,
    mode: str,
    *,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    random_state: int = 0,
    max_iter: int = 500,
    n_init: int = 5,
    weight_concentration_prior: float | None = None,
) -> GaussianMixture:
    mode = (mode or "gaussian").lower()
    if mode == "bayesian":
        prior = weight_concentration_prior
        if prior is None:
            prior = 1.0
        model = BayesianGaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=prior,
            max_iter=max_iter,
            init_params="kmeans",
            random_state=random_state,
        )
    else:
        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            init_params="kmeans",
        )
    model.fit(X)
    return model


def get_active_components(
    model: GaussianMixture | BayesianGaussianMixture,
    weight_threshold: float = 0.01,
) -> Tuple[int, np.ndarray]:
    """Get number of active components and their weights.

    For BGMM with DP prior, components with weights below threshold are considered inactive.

    Returns:
        (n_active, active_weights): Number of active components and their normalized weights
    """
    weights = model.weights_
    active_mask = weights >= weight_threshold
    n_active = int(np.sum(active_mask))
    active_weights = weights[active_mask]

    # Renormalize active weights
    if active_weights.sum() > 0:
        active_weights = active_weights / active_weights.sum()

    return n_active, active_weights


def fit_gmm_BICICL(
    X: Array,
    k_grid: Tuple[int, ...] = (2, 3, 4, 5, 6),
    *,
    mode: str = "bayesian",
    random_state: int = 0,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    max_iter: int = 500,
    n_init: int = 5,
    weight_concentration_prior: float | None = None,
    weight_threshold: float = 0.01,
) -> Tuple[Tuple[int, float, object], List[Tuple[int, float, float, object]]]:
    """Fit Gaussian/Bayesian GMMs and select with ICL.

    Returns best (k, icl, model) and a table of all fits with (k, bic, icl, model).

    For Bayesian mode with DP prior:
    - Automatically detects active components based on weight_threshold
    - Reports both nominal K and effective K_active
    """

    best: Tuple[int, float, object] | None = None
    table: List[Tuple[int, float, float, object]] = []

    for k in k_grid:
        gm = _fit_gaussian_mixture(
            X,
            k,
            mode,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            weight_concentration_prior=weight_concentration_prior,
        )

        try:
            bic = float(gm.bic(X))
        except AttributeError:
            # BayesianGaussianMixture before sklearn 1.1 lacks bic(); approximate with log-likelihood
            log_likelihood = float(gm.score(X) * X.shape[0])
            bic = -2.0 * log_likelihood + k * np.log(X.shape[0])

        resp = gm.predict_proba(X)
        eps = 1e-12
        entropy_penalty = float(np.sum(resp * np.log(resp + eps)))
        icl = bic - entropy_penalty

        table.append((k, bic, icl, gm))
        if best is None or icl < best[1]:
            best = (k, icl, gm)

    assert best is not None
    return best, table


__all__ = ["fit_gmm_BICICL", "_fit_gaussian_mixture", "get_active_components"]
