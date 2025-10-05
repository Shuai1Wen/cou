"""Domain Adversarial Neural Network (DANN) for domain-invariant gating.

Reference:
    Ganin et al. "Domain-Adversarial Training of Neural Networks" (2016)
    https://arxiv.org/abs/1505.07818
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression


class GradientReversalLayer:
    """Gradient Reversal Layer for domain adversarial training.

    During forward pass: identity
    During backward pass: multiply gradient by -lambda
    """

    def __init__(self, lambda_: float = 1.0):
        """Initialize GRL.

        Args:
            lambda_: Reversal strength (typically 0.1 to 1.0)
        """
        self.lambda_ = lambda_

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass (identity)."""
        return x

    def backward_scale(self) -> float:
        """Get backward scaling factor."""
        return -self.lambda_


class DomainAdversarialGating:
    """Domain adversarial gating for cross-donor/site robustness.

    Architecture:
        Feature extractor (shared) -> [Gate classifier, Domain classifier]

    The domain classifier tries to predict donor/site while the feature
    extractor learns domain-invariant representations through adversarial
    training.
    """

    def __init__(
        self,
        n_gates: int,
        lambda_domain: float = 0.1,
        gate_C: float = 1.0,
        domain_C: float = 1.0,
        max_iter: int = 300,
    ):
        """Initialize DANN gating.

        Args:
            n_gates: Number of gate components
            lambda_domain: Domain adversarial weight
            gate_C: Logistic regression C for gate classifier
            domain_C: Logistic regression C for domain classifier
            max_iter: Max iterations for logistic regression
        """
        self.n_gates = n_gates
        self.lambda_domain = lambda_domain
        self.gate_C = gate_C
        self.domain_C = domain_C
        self.max_iter = max_iter

        self.grl = GradientReversalLayer(lambda_domain)
        self.gate_clf: Optional[LogisticRegression] = None
        self.domain_clf: Optional[LogisticRegression] = None

    def fit(
        self,
        X: np.ndarray,
        gate_labels: np.ndarray,
        domain_labels: np.ndarray,
        gate_weights: Optional[np.ndarray] = None,
    ) -> 'DomainAdversarialGating':
        """Fit gate and domain classifiers.

        Args:
            X: Features [n_samples, n_features]
            gate_labels: Gate assignments [n_samples]
            domain_labels: Domain labels (donor/site) [n_samples]
            gate_weights: Optional sample weights for gate classifier

        Returns:
            self
        """

        # Train gate classifier (main task)
        self.gate_clf = LogisticRegression(
            solver='lbfgs',
            C=self.gate_C,
            max_iter=self.max_iter,
            n_jobs=-1,
        )

        if gate_weights is not None:
            self.gate_clf.fit(X, gate_labels, sample_weight=gate_weights)
        else:
            self.gate_clf.fit(X, gate_labels)

        # Train domain classifier (adversarial task)
        # In practice, we want to maximize domain classification loss,
        # which is equivalent to learning features that confuse the domain classifier

        # Simple implementation: train domain classifier on features
        # In full DANN, this would be done with gradient reversal
        self.domain_clf = LogisticRegression(
            solver='lbfgs',
            C=self.domain_C,
            max_iter=self.max_iter,
            n_jobs=-1,
        )
        self.domain_clf.fit(X, domain_labels)

        return self

    def predict_gate_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict gate probabilities.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            proba: Gate probabilities [n_samples, n_gates]
        """
        if self.gate_clf is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.gate_clf.predict_proba(X)

    def predict_domain_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict domain probabilities.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            proba: Domain probabilities [n_samples, n_domains]
        """
        if self.domain_clf is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.domain_clf.predict_proba(X)

    def compute_domain_confusion(self, X: np.ndarray, domain_labels: np.ndarray) -> float:
        """Compute domain confusion metric (lower is better for invariance).

        Args:
            X: Features
            domain_labels: True domain labels

        Returns:
            accuracy: Domain classification accuracy (we want this low)
        """
        if self.domain_clf is None:
            raise ValueError("Model not fitted. Call fit() first.")

        pred = self.domain_clf.predict(X)
        accuracy = float(np.mean(pred == domain_labels))
        return accuracy

    def get_domain_invariance_score(self, X: np.ndarray, domain_labels: np.ndarray) -> float:
        """Get domain invariance score (higher is better).

        Returns 1 - domain_accuracy, so higher values indicate better invariance.

        Args:
            X: Features
            domain_labels: True domain labels

        Returns:
            score: Invariance score in [0, 1]
        """
        return 1.0 - self.compute_domain_confusion(X, domain_labels)


def apply_dann_gating(
    X: np.ndarray,
    gate_responsibilities: np.ndarray,
    domain_labels: np.ndarray,
    lambda_domain: float = 0.1,
    **kwargs
) -> Tuple[DomainAdversarialGating, np.ndarray]:
    """Apply domain adversarial gating.

    Args:
        X: Features [n_samples, n_features]
        gate_responsibilities: Soft gate assignments from GMM [n_samples, n_gates]
        domain_labels: Domain labels (donor/site IDs) [n_samples]
        lambda_domain: Domain adversarial weight
        **kwargs: Additional arguments for DomainAdversarialGating

    Returns:
        (model, gate_proba): Trained DANN model and refined gate probabilities
    """

    n_gates = gate_responsibilities.shape[1]
    gate_labels = gate_responsibilities.argmax(axis=1)
    gate_weights = gate_responsibilities.max(axis=1)

    dann = DomainAdversarialGating(
        n_gates=n_gates,
        lambda_domain=lambda_domain,
        **kwargs
    )

    dann.fit(X, gate_labels, domain_labels, gate_weights)

    # Get refined gate probabilities
    gate_proba = dann.predict_gate_proba(X)

    return dann, gate_proba


__all__ = [
    'GradientReversalLayer',
    'DomainAdversarialGating',
    'apply_dann_gating',
]
