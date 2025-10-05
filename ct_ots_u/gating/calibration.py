#!/usr/bin/env python3
"""Temperature scaling and calibration for gating models.

Implements temperature scaling to calibrate gate probabilities on target domain,
following "On Calibration of Modern Neural Networks" (Guo et al., ICML 2017).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss


def nll_with_temperature(
    T: float,
    logits: np.ndarray,
    y_true: np.ndarray | None = None,
) -> float:
    """Compute negative log-likelihood with temperature scaling.

    Args:
        T: Temperature parameter
        logits: Raw logits [n_samples, n_classes]
        y_true: True labels [n_samples]. If None, use argmax of scaled probs as pseudo-labels

    Returns:
        nll: Negative log-likelihood
    """
    T = np.clip(T, 1e-3, 100.0)

    # Temperature-scaled probabilities
    scaled_logits = logits / T
    scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)  # Numerical stability
    probs = np.exp(scaled_logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Use pseudo-labels if no true labels
    if y_true is None:
        y_true = probs.argmax(axis=1)

    # Compute NLL
    nll = -np.log(probs[np.arange(len(probs)), y_true] + 1e-12).mean()
    return float(nll)


def calibrate_temperature(
    logits: np.ndarray,
    y_true: np.ndarray | None = None,
    method: str = 'scalar',
    bounds: Tuple[float, float] = (0.1, 10.0),
) -> dict:
    """Calibrate temperature parameter on validation/target data.

    Args:
        logits: Raw logits [n_samples, n_classes]
        y_true: True labels [n_samples]. If None, use pseudo-labels
        method: 'scalar' (single T) or 'vector' (per-class T, not implemented)
        bounds: Temperature search bounds

    Returns:
        calibration_dict with keys:
            - temperature: Optimal T
            - nll_before: NLL before scaling
            - nll_after: NLL after scaling
            - method: Calibration method
    """
    if method != 'scalar':
        raise NotImplementedError("Only scalar temperature scaling is implemented")

    # NLL before scaling (T=1)
    nll_before = nll_with_temperature(1.0, logits, y_true)

    # Optimize T
    result = minimize_scalar(
        lambda t: nll_with_temperature(t, logits, y_true),
        bounds=bounds,
        method='bounded',
    )

    T_opt = float(result.x)
    nll_after = float(result.fun)

    return {
        'temperature': T_opt,
        'nll_before': nll_before,
        'nll_after': nll_after,
        'nll_improvement': nll_before - nll_after,
        'method': method,
    }


def compute_ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities [n_samples, n_classes]
        y_true: True labels [n_samples]
        n_bins: Number of bins for calibration curve

    Returns:
        ece: Expected Calibration Error
    """
    y_pred = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    accuracies = (y_pred == y_true).astype(float)

    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def apply_temperature_scaling(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply temperature scaling to logits.

    Args:
        logits: Raw logits [n_samples, n_classes]
        temperature: Temperature parameter

    Returns:
        probs: Temperature-scaled probabilities [n_samples, n_classes]
    """
    scaled_logits = logits / temperature
    scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    probs = np.exp(scaled_logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def evaluate_calibration(
    logits: np.ndarray,
    y_true: np.ndarray,
    temperature: float = 1.0,
    n_bins: int = 15,
) -> dict:
    """Evaluate calibration metrics.

    Args:
        logits: Raw logits [n_samples, n_classes]
        y_true: True labels [n_samples]
        temperature: Temperature parameter
        n_bins: Number of ECE bins

    Returns:
        metrics_dict with ECE, NLL, accuracy
    """
    # Get probabilities
    probs = apply_temperature_scaling(logits, temperature)

    # Compute metrics
    ece = compute_ece(probs, y_true, n_bins)
    nll = log_loss(y_true, probs)
    acc = (probs.argmax(axis=1) == y_true).mean()

    return {
        'ece': float(ece),
        'nll': float(nll),
        'accuracy': float(acc),
        'temperature': float(temperature),
    }


__all__ = [
    'calibrate_temperature',
    'apply_temperature_scaling',
    'compute_ece',
    'evaluate_calibration',
    'nll_with_temperature',
]
