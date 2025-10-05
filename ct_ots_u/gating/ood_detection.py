#!/usr/bin/env python3
"""Out-of-distribution detection for gating using Mahalanobis distance.

Implements Mahalanobis-based OOD detection in gating embedding space,
following Lee et al., "A Simple Unified Framework for Detecting OOD Samples
and Adversarial Attacks" (NeurIPS 2018).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.linalg import inv
from sklearn.covariance import EmpiricalCovariance, LedoitWolf


class MahalanobisOOD:
    """Mahalanobis distance-based OOD detector for gate embeddings.

    Fits a Gaussian distribution for each gate in embedding space,
    then computes Mahalanobis distance to detect out-of-distribution samples.
    """

    def __init__(
        self,
        covariance_estimator: str = 'empirical',
        shrinkage: bool = True,
    ):
        """Initialize OOD detector.

        Args:
            covariance_estimator: 'empirical' or 'ledoit_wolf'
            shrinkage: Whether to apply shrinkage for numerical stability
        """
        self.covariance_estimator = covariance_estimator
        self.shrinkage = shrinkage

        self.means_ = None
        self.precisions_ = None
        self.n_gates_ = None

    def fit(
        self,
        X: np.ndarray,
        gate_labels: np.ndarray,
    ) -> 'MahalanobisOOD':
        """Fit Gaussian distributions for each gate.

        Args:
            X: Gate embeddings [n_samples, n_features]
            gate_labels: Gate assignments [n_samples]

        Returns:
            self
        """
        unique_gates = np.unique(gate_labels)
        self.n_gates_ = len(unique_gates)

        self.means_ = []
        self.precisions_ = []

        for gate in unique_gates:
            mask = (gate_labels == gate)
            X_gate = X[mask]

            # Compute mean
            mean = X_gate.mean(axis=0)
            self.means_.append(mean)

            # Compute precision matrix (inverse covariance)
            if self.covariance_estimator == 'ledoit_wolf':
                cov_estimator = LedoitWolf().fit(X_gate)
                cov = cov_estimator.covariance_
            else:
                cov = np.cov(X_gate.T)

            # Add regularization for numerical stability
            if self.shrinkage:
                cov = cov + 1e-6 * np.eye(cov.shape[0])

            precision = inv(cov)
            self.precisions_.append(precision)

        return self

    def score(
        self,
        X: np.ndarray,
        return_gate: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Compute minimum Mahalanobis distance to any gate.

        Args:
            X: Embeddings [n_samples, n_features]
            return_gate: If True, also return closest gate index

        Returns:
            scores: Minimum Mahalanobis distances [n_samples]
            gates: (optional) Closest gate indices [n_samples]
        """
        if self.means_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_samples = X.shape[0]
        min_distances = np.full(n_samples, np.inf)
        closest_gates = np.zeros(n_samples, dtype=int)

        for gate_idx, (mean, precision) in enumerate(zip(self.means_, self.precisions_)):
            # Compute Mahalanobis distance for all samples to this gate
            diff = X - mean[None, :]  # [n_samples, n_features]
            mahal = np.sqrt(np.sum(diff @ precision * diff, axis=1))

            # Update minimum distances
            update_mask = mahal < min_distances
            min_distances[update_mask] = mahal[update_mask]
            closest_gates[update_mask] = gate_idx

        if return_gate:
            return min_distances, closest_gates
        return min_distances

    def reject(
        self,
        X: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Reject samples above distance threshold as OOD.

        Args:
            X: Embeddings [n_samples, n_features]
            threshold: Distance threshold (e.g., 95th percentile from validation)

        Returns:
            reject_mask: Boolean mask [n_samples] (True = OOD/rejected)
        """
        distances = self.score(X)
        return distances > threshold

    def predict_with_rejection(
        self,
        X: np.ndarray,
        gate_probs: np.ndarray,
        threshold: float,
        rejection_label: int = -1,
    ) -> np.ndarray:
        """Predict gates with OOD rejection.

        Args:
            X: Embeddings [n_samples, n_features]
            gate_probs: Gate probabilities [n_samples, n_gates]
            threshold: Distance threshold
            rejection_label: Label for rejected samples

        Returns:
            predictions: Gate predictions with rejections [n_samples]
        """
        distances, closest_gates = self.score(X, return_gate=True)
        is_ood = distances > threshold

        predictions = gate_probs.argmax(axis=1)
        predictions[is_ood] = rejection_label

        return predictions


def calibrate_ood_threshold(
    detector: MahalanobisOOD,
    X_val: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Calibrate OOD threshold on validation set.

    Args:
        detector: Fitted MahalanobisOOD detector
        X_val: Validation embeddings
        percentile: Percentile for threshold (e.g., 95 = reject top 5%)

    Returns:
        threshold: Distance threshold
    """
    scores = detector.score(X_val)
    threshold = float(np.percentile(scores, percentile))
    return threshold


def evaluate_ood_detection(
    detector: MahalanobisOOD,
    X_id: np.ndarray,
    X_ood: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate OOD detection performance.

    Args:
        detector: Fitted detector
        X_id: In-distribution samples
        X_ood: Out-of-distribution samples
        threshold: Decision threshold

    Returns:
        metrics_dict with TPR, FPR, AUROC
    """
    from sklearn.metrics import roc_auc_score

    # Compute scores
    scores_id = detector.score(X_id)
    scores_ood = detector.score(X_ood)

    # Binary labels (0 = ID, 1 = OOD)
    y_true = np.concatenate([
        np.zeros(len(scores_id)),
        np.ones(len(scores_ood)),
    ])
    y_scores = np.concatenate([scores_id, scores_ood])

    # AUROC
    auroc = roc_auc_score(y_true, y_scores)

    # TPR/FPR at threshold
    tpr = (scores_ood > threshold).mean()
    fpr = (scores_id > threshold).mean()

    return {
        'auroc': float(auroc),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'threshold': float(threshold),
    }


__all__ = [
    'MahalanobisOOD',
    'calibrate_ood_threshold',
    'evaluate_ood_detection',
]
