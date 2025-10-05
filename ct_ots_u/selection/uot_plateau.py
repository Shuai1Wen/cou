"""UOT hyperparameter plateau selection with 1-SE rule."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def compute_1se_threshold(
    mean: float,
    std: float,
    n: int,
    confidence: float = 0.68
) -> float:
    """Compute 1-SE threshold using standard error.

    Args:
        mean: Mean error
        std: Standard deviation
        n: Number of samples
        confidence: Confidence level (0.68 for 1-SE)

    Returns:
        threshold: mean + SE
    """
    se = std / np.sqrt(n)
    return mean + se


def select_plateau_point(
    param_grid: List[Tuple],
    errors: Dict[Tuple, List[float]],
    param_names: List[str] = ['tau', 'eps', 'reg_m'],
    one_se: bool = True,
    prefer_smaller_tau: bool = True,
) -> Tuple[Tuple, Dict]:
    """Select UOT hyperparameter using plateau selection with 1-SE rule.

    Strategy:
    1. Find configuration with minimum mean error
    2. Compute 1-SE threshold
    3. Among all configs within 1-SE band, select:
       - If prefer_smaller_tau: smallest tau (reduces extrapolation bias)
       - Otherwise: config with smallest error

    Args:
        param_grid: List of parameter tuples [(tau1, eps1, reg_m1), ...]
        errors: Dict mapping param tuple to list of error values
        param_names: Names of parameters
        one_se: Use 1-SE rule
        prefer_smaller_tau: Prefer smaller tau in plateau

    Returns:
        (best_params, stats): Best parameter tuple and statistics dict
    """

    # Compute statistics for each config
    stats_dict = {}
    for params in param_grid:
        err_list = errors[params]
        stats_dict[params] = {
            'mean': np.mean(err_list),
            'std': np.std(err_list),
            'se': np.std(err_list) / np.sqrt(len(err_list)),
            'n': len(err_list),
            'values': err_list,
        }

    # Find best (minimum mean error)
    best_params = min(param_grid, key=lambda p: stats_dict[p]['mean'])
    best_mean = stats_dict[best_params]['mean']
    best_std = stats_dict[best_params]['std']
    best_n = stats_dict[best_params]['n']

    if not one_se:
        return best_params, {
            'best_params': dict(zip(param_names, best_params)),
            'best_mean': best_mean,
            'best_std': best_std,
            'threshold': None,
            'plateau_candidates': [],
            'selection_mode': 'min_error',
        }

    # Compute 1-SE threshold
    threshold = compute_1se_threshold(best_mean, best_std, best_n)

    # Find all configs within 1-SE band
    plateau_candidates = [
        p for p in param_grid
        if stats_dict[p]['mean'] <= threshold
    ]

    # Select from plateau
    if prefer_smaller_tau and param_names[0] == 'tau':
        # Select smallest tau among plateau candidates
        selected = min(plateau_candidates, key=lambda p: p[0])
        selection_mode = 'min_tau_in_1se'
    else:
        # Select minimum error in plateau
        selected = min(plateau_candidates, key=lambda p: stats_dict[p]['mean'])
        selection_mode = 'min_error_in_1se'

    return selected, {
        'best_params': dict(zip(param_names, best_params)),
        'selected_params': dict(zip(param_names, selected)),
        'best_mean': best_mean,
        'best_std': best_std,
        'selected_mean': stats_dict[selected]['mean'],
        'selected_std': stats_dict[selected]['std'],
        'threshold': threshold,
        'plateau_size': len(plateau_candidates),
        'plateau_candidates': [
            dict(zip(param_names, p)) for p in plateau_candidates
        ],
        'selection_mode': selection_mode,
        'all_stats': {
            str(dict(zip(param_names, p))): {
                'mean': stats_dict[p]['mean'],
                'std': stats_dict[p]['std'],
                'se': stats_dict[p]['se'],
            }
            for p in param_grid
        }
    }


def evaluate_uot_grid(
    source_data: np.ndarray,
    target_data: np.ndarray,
    tau_grid: List[float],
    eps_grid: List[float],
    reg_m_grid: List[float],
    n_repeats: int = 3,
    metric_fn=None,
    random_state: int = 42,
) -> Tuple[Dict[Tuple, List[float]], List[Tuple]]:
    """Evaluate UOT distance on parameter grid with bootstrap repeats.

    Args:
        source_data: Source samples [n_source, d]
        target_data: Target samples [n_target, d]
        tau_grid: Time step candidates
        eps_grid: Entropic regularization candidates
        reg_m_grid: Marginal relaxation candidates
        n_repeats: Number of bootstrap repeats
        metric_fn: Optional metric function(X, Y, tau, eps, reg_m) -> float
        random_state: Random seed

    Returns:
        (errors_dict, param_grid): Errors for each config and parameter grid
    """

    if metric_fn is None:
        # Default: use POT unbalanced Sinkhorn
        from ..ot_metrics import compute_uot_distance
        metric_fn = lambda X, Y, tau, eps, reg_m: compute_uot_distance(
            X, Y, eps=eps, reg_m=reg_m
        )[0]

    rng = np.random.default_rng(random_state)
    param_grid = [
        (tau, eps, reg_m)
        for tau in tau_grid
        for eps in eps_grid
        for reg_m in reg_m_grid
    ]

    errors = {params: [] for params in param_grid}

    n_source = source_data.shape[0]
    n_target = target_data.shape[0]

    for repeat in range(n_repeats):
        # Bootstrap sample
        idx_s = rng.choice(n_source, size=n_source, replace=True)
        idx_t = rng.choice(n_target, size=n_target, replace=True)

        X_boot = source_data[idx_s]
        Y_boot = target_data[idx_t]

        for tau, eps, reg_m in param_grid:
            try:
                error = metric_fn(X_boot, Y_boot, tau, eps, reg_m)
                errors[(tau, eps, reg_m)].append(float(error))
            except Exception as e:
                print(f"Warning: Failed for ({tau}, {eps}, {reg_m}): {e}")
                errors[(tau, eps, reg_m)].append(float('inf'))

    return errors, param_grid


__all__ = [
    'select_plateau_point',
    'evaluate_uot_grid',
    'compute_1se_threshold',
]
