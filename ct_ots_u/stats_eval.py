"""Statistical utilities for final validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from scipy import stats


def bootstrap_ci(
    func: Callable[[np.ndarray, np.ndarray], float],
    X: np.ndarray,
    Y: np.ndarray,
    n_resamples: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    idx_x = np.arange(X.shape[0])
    idx_y = np.arange(Y.shape[0])
    values = []
    for _ in range(n_resamples):
        bx = X[rng.choice(idx_x, size=idx_x.size, replace=True)]
        by = Y[rng.choice(idx_y, size=idx_y.size, replace=True)]
        values.append(func(bx, by))
    arr = np.asarray(values)
    return float(arr.mean()), (float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2)))


def paired_test(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    sample_a = np.asarray(sample_a)
    sample_b = np.asarray(sample_b)
    sample_a = sample_a.reshape(-1)
    sample_b = sample_b.reshape(-1)
    stat, p_value = stats.ttest_rel(sample_a, sample_b, nan_policy="omit")
    return float(p_value)


def dump_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
