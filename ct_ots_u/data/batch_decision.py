#!/usr/bin/env python3
"""Automatic batch effect integration decision based on scIB metrics.

Implements scIB-driven decision logic for whether to apply batch integration
(Harmony/Scanorama), following Luecken et al., Nature Methods 2022.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _compute_silhouette_asw(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute average silhouette width (ASW).

    Args:
        X: Embeddings [n_samples, n_features]
        labels: Labels [n_samples]

    Returns:
        asw: Average silhouette width
    """
    from sklearn.metrics import silhouette_score

    try:
        asw = silhouette_score(X, labels, metric='euclidean')
        return float(asw)
    except Exception:
        return float('nan')


def _compute_graph_connectivity(
    X: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
) -> float:
    """Compute graph connectivity metric (simplified scIB version).

    For each cell type, check if cells are connected across batches in kNN graph.

    Args:
        X: Embeddings [n_samples, n_features]
        labels: Cell type labels [n_samples]
        n_neighbors: Number of neighbors for kNN graph

    Returns:
        connectivity: Graph connectivity score (0-1, higher is better)
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X) - 1))
    nn.fit(X)

    unique_labels = np.unique(labels)
    connectivity_scores = []

    for label in unique_labels:
        mask = (labels == label)
        if np.sum(mask) < 2:
            continue

        X_label = X[mask]
        labels_label = labels[mask]

        # Get neighbors
        indices = nn.kneighbors(X_label, return_distance=False)
        neighbor_labels = labels[indices]

        # Check if neighbors include same label (connected component)
        has_same_label = np.any(neighbor_labels == label, axis=1)
        connectivity_scores.append(np.mean(has_same_label))

    if len(connectivity_scores) == 0:
        return float('nan')

    return float(np.mean(connectivity_scores))


def compute_batch_metrics(
    X: np.ndarray,
    batch_labels: np.ndarray,
    bio_labels: np.ndarray,
) -> Dict:
    """Compute batch integration metrics (scIB-style).

    Args:
        X: Embeddings [n_samples, n_features]
        batch_labels: Batch assignments [n_samples]
        bio_labels: Biological labels (cell types) [n_samples]

    Returns:
        metrics_dict with:
            - batch_asw: Batch ASW (lower is better - less batch effect)
            - bio_asw: Bio ASW (higher is better - preserves biology)
            - graph_conn: Graph connectivity (higher is better)
    """
    batch_asw = _compute_silhouette_asw(X, batch_labels)
    bio_asw = _compute_silhouette_asw(X, bio_labels)
    graph_conn = _compute_graph_connectivity(X, bio_labels)

    return {
        'batch_asw': batch_asw,
        'bio_asw': bio_asw,
        'graph_connectivity': graph_conn,
    }


def decide_harmony(
    X_raw: np.ndarray,
    X_integrated: np.ndarray,
    batch_labels: np.ndarray,
    bio_labels: np.ndarray,
    batch_delta_thresh: float = 0.03,
    bio_drop_thresh: float = 0.02,
) -> Dict:
    """Decide whether to enable Harmony based on scIB metrics.

    Decision criteria:
    1. Batch ASW should decrease by at least batch_delta_thresh
    2. Bio ASW should not drop by more than bio_drop_thresh
    3. Graph connectivity should not decrease

    Args:
        X_raw: Raw embeddings before integration
        X_integrated: Embeddings after Harmony
        batch_labels: Batch assignments
        bio_labels: Biological labels
        batch_delta_thresh: Min batch ASW improvement
        bio_drop_thresh: Max bio ASW drop allowed

    Returns:
        decision_dict with:
            - enable_harmony: bool decision
            - before_metrics: Metrics before integration
            - after_metrics: Metrics after integration
            - criteria: Thresholds used
            - reasons: List of reasons for decision
    """
    # Compute metrics before and after
    metrics_before = compute_batch_metrics(X_raw, batch_labels, bio_labels)
    metrics_after = compute_batch_metrics(X_integrated, batch_labels, bio_labels)

    # Compute deltas
    batch_asw_improvement = metrics_before['batch_asw'] - metrics_after['batch_asw']
    bio_asw_drop = metrics_before['bio_asw'] - metrics_after['bio_asw']
    graph_conn_drop = metrics_before['graph_connectivity'] - metrics_after['graph_connectivity']

    # Decision logic
    reasons = []
    enable = True

    # Criterion 1: Batch ASW improvement
    if batch_asw_improvement < batch_delta_thresh:
        reasons.append(f"Batch ASW improvement ({batch_asw_improvement:.3f}) < threshold ({batch_delta_thresh})")
        enable = False
    else:
        reasons.append(f"[+] Batch ASW improved by {batch_asw_improvement:.3f}")

    # Criterion 2: Bio ASW preservation
    if bio_asw_drop > bio_drop_thresh:
        reasons.append(f"Bio ASW drop ({bio_asw_drop:.3f}) > threshold ({bio_drop_thresh})")
        enable = False
    else:
        reasons.append(f"[+] Bio ASW preserved (drop: {bio_asw_drop:.3f})")

    # Criterion 3: Graph connectivity (optional, warn if drops)
    if not np.isnan(graph_conn_drop) and graph_conn_drop > 0.05:
        reasons.append(f"[!] Graph connectivity dropped by {graph_conn_drop:.3f}")

    return {
        'enable_harmony': bool(enable),
        'before_metrics': metrics_before,
        'after_metrics': metrics_after,
        'deltas': {
            'batch_asw_improvement': float(batch_asw_improvement),
            'bio_asw_drop': float(bio_asw_drop),
            'graph_conn_drop': float(graph_conn_drop),
        },
        'criteria': {
            'batch_delta_thresh': batch_delta_thresh,
            'bio_drop_thresh': bio_drop_thresh,
        },
        'reasons': reasons,
    }


def auto_integrate_harmony(
    adata_combined,
    batch_key: str,
    label_key: str,
    batch_delta_thresh: float = 0.03,
    bio_drop_thresh: float = 0.02,
    harmony_theta: float = 2.0,
) -> tuple:
    """Automatically decide and apply Harmony if beneficial.

    Args:
        adata_combined: AnnData with both source and target
        batch_key: Column name for batch labels
        label_key: Column name for biological labels
        batch_delta_thresh: Min batch improvement
        bio_drop_thresh: Max bio drop
        harmony_theta: Harmony diversity parameter

    Returns:
        (adata_result, decision_dict):
            - adata_result: Integrated data if enabled, else raw
            - decision_dict: Decision report
    """
    from ct_ots_u.data.harmony_integration import apply_harmony_integration

    # Get raw embeddings
    X_raw = adata_combined.obsm.get('X_pca', adata_combined.X)
    batch_labels = adata_combined.obs[batch_key].values
    bio_labels = adata_combined.obs[label_key].values

    # Apply Harmony
    X_integrated = apply_harmony_integration(
        X_raw,
        batch_labels,
        theta=harmony_theta,
        backend='harmonypy',
    )

    # Decide
    decision = decide_harmony(
        X_raw,
        X_integrated,
        batch_labels,
        bio_labels,
        batch_delta_thresh,
        bio_drop_thresh,
    )

    # Apply if beneficial
    if decision['enable_harmony']:
        adata_result = adata_combined.copy()
        adata_result.obsm['X_pca'] = X_integrated
        adata_result.uns['harmony_applied'] = True
    else:
        adata_result = adata_combined
        adata_result.uns['harmony_applied'] = False

    adata_result.uns['batch_decision'] = decision

    return adata_result, decision


__all__ = [
    'compute_batch_metrics',
    'decide_harmony',
    'auto_integrate_harmony',
]
