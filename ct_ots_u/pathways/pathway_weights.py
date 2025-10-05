"""Pathway-based distance metrics for CT-OTS-U."""

from __future__ import annotations

from typing import Dict, Set, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

Array = np.ndarray


def compute_pathway_weights(
    gene_names: List[str],
    gene_sets: Dict[str, Set[str]],
    weighting_method: str = "uniform",
    pathway_importance: Optional[Dict[str, float]] = None
) -> Array:
    """Compute pathway-based gene importance weights.

    Parameters
    ----------
    gene_names : List[str]
        List of gene names in order
    gene_sets : Dict[str, Set[str]]
        Dictionary of pathway gene sets
    weighting_method : str
        Method for computing weights ('uniform', 'inverse_size', 'importance')
    pathway_importance : Dict[str, float], optional
        Manual pathway importance scores

    Returns
    -------
    weights : Array
        Gene weights of shape (n_genes,)
    """

    n_genes = len(gene_names)
    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    weights = np.zeros(n_genes)

    for pathway_name, pathway_genes in gene_sets.items():
        # Find genes present in our data
        present_genes = [gene for gene in pathway_genes if gene in gene_to_idx]

        if not present_genes:
            continue

        # Compute pathway weight
        if weighting_method == "uniform":
            pathway_weight = 1.0
        elif weighting_method == "inverse_size":
            pathway_weight = 1.0 / np.sqrt(len(pathway_genes))
        elif weighting_method == "importance":
            pathway_weight = pathway_importance.get(pathway_name, 1.0) if pathway_importance else 1.0
        else:
            raise ValueError(f"Unknown weighting method: {weighting_method}")

        # Add weight to genes in this pathway
        for gene in present_genes:
            idx = gene_to_idx[gene]
            weights[idx] += pathway_weight

    # Normalize weights
    if weights.sum() > 0:
        zero_mask = weights == 0
        if np.any(zero_mask):
            positive = weights[weights > 0]
            baseline = float(positive.min()) if positive.size else 1.0
            if baseline <= 0:
                baseline = 1.0
            baseline *= 0.1
            weights[zero_mask] = baseline
        weights = weights / weights.sum() * n_genes
    else:
        weights = np.ones(n_genes)  # Fallback to uniform weights



    return weights


def create_pathway_metric(
    gene_names: List[str],
    gene_sets: Dict[str, Set[str]],
    base_metric: str = "euclidean",
    pathway_weight: float = 0.5,
    weighting_method: str = "uniform"
) -> callable:
    """Create a pathway-weighted distance metric.

    Parameters
    ----------
    gene_names : List[str]
        Gene names corresponding to feature dimensions
    gene_sets : Dict[str, Set[str]]
        Pathway gene sets
    base_metric : str
        Base distance metric
    pathway_weight : float
        Weight for pathway information (0-1)
    weighting_method : str
        Method for computing pathway weights

    Returns
    -------
    metric_function : callable
        Distance function that takes two arrays and returns distances
    """

    weights = compute_pathway_weights(gene_names, gene_sets, weighting_method)

    def pathway_weighted_metric(X: Array, Y: Array) -> Array:
        """Compute pathway-weighted distances between point sets.

        Parameters
        ----------
        X : Array
            First point set of shape (n, d)
        Y : Array
            Second point set of shape (m, d)

        Returns
        -------
        distances : Array
            Distance matrix of shape (n, m)
        """

        # Standard distance
        if base_metric == "euclidean":
            base_distances = cdist(X, Y, metric="euclidean")
        elif base_metric == "sqeuclidean":
            base_distances = cdist(X, Y, metric="sqeuclidean")
        else:
            base_distances = cdist(X, Y, metric=base_metric)

        # Weighted distance
        if pathway_weight > 0 and np.any(weights != weights[0]):  # Non-uniform weights
            # Apply weights to features
            X_weighted = X * np.sqrt(weights)
            Y_weighted = Y * np.sqrt(weights)

            if base_metric == "euclidean":
                weighted_distances = cdist(X_weighted, Y_weighted, metric="euclidean")
            elif base_metric == "sqeuclidean":
                weighted_distances = cdist(X_weighted, Y_weighted, metric="sqeuclidean")
            else:
                weighted_distances = cdist(X_weighted, Y_weighted, metric=base_metric)

            # Combine base and weighted distances
            final_distances = (1 - pathway_weight) * base_distances + pathway_weight * weighted_distances
        else:
            final_distances = base_distances

        return final_distances

    return pathway_weighted_metric


def pathway_gene_overlap_matrix(
    gene_names: List[str],
    gene_sets: Dict[str, Set[str]]
) -> Tuple[Array, List[str]]:
    """Create gene-pathway membership matrix.

    Parameters
    ----------
    gene_names : List[str]
        Gene names
    gene_sets : Dict[str, Set[str]]
        Pathway gene sets

    Returns
    -------
    membership_matrix : Array
        Binary matrix of shape (n_genes, n_pathways)
    pathway_names : List[str]
        Pathway names corresponding to columns
    """

    n_genes = len(gene_names)
    pathway_names = list(gene_sets.keys())
    n_pathways = len(pathway_names)

    gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
    membership_matrix = np.zeros((n_genes, n_pathways), dtype=bool)

    for j, (pathway_name, pathway_genes) in enumerate(gene_sets.items()):
        for gene in pathway_genes:
            if gene in gene_to_idx:
                i = gene_to_idx[gene]
                membership_matrix[i, j] = True

    return membership_matrix, pathway_names


def compute_pathway_similarity(
    gene_sets: Dict[str, Set[str]],
    method: str = "jaccard"
) -> Tuple[Array, List[str]]:
    """Compute pairwise pathway similarity matrix.

    Parameters
    ----------
    gene_sets : Dict[str, Set[str]]
        Pathway gene sets
    method : str
        Similarity method ('jaccard', 'overlap')

    Returns
    -------
    similarity_matrix : Array
        Pathway similarity matrix
    pathway_names : List[str]
        Pathway names
    """

    pathway_names = list(gene_sets.keys())
    n_pathways = len(pathway_names)
    similarity_matrix = np.zeros((n_pathways, n_pathways))

    for i, pathway_i in enumerate(pathway_names):
        genes_i = gene_sets[pathway_i]
        for j, pathway_j in enumerate(pathway_names):
            genes_j = gene_sets[pathway_j]

            intersection = genes_i & genes_j
            union = genes_i | genes_j

            if method == "jaccard":
                similarity = len(intersection) / len(union) if union else 0
            elif method == "overlap":
                similarity = len(intersection) / min(len(genes_i), len(genes_j)) if min(len(genes_i), len(genes_j)) > 0 else 0
            else:
                raise ValueError(f"Unknown similarity method: {method}")

            similarity_matrix[i, j] = similarity

    return similarity_matrix, pathway_names


def select_representative_pathways(
    gene_sets: Dict[str, Set[str]],
    max_pathways: int = 100,
    min_size: int = 15,
    max_size: int = 500,
    similarity_threshold: float = 0.7
) -> Dict[str, Set[str]]:
    """Select representative pathways to reduce redundancy.

    Parameters
    ----------
    gene_sets : Dict[str, Set[str]]
        Input pathway gene sets
    max_pathways : int
        Maximum number of pathways to select
    min_size, max_size : int
        Size constraints for pathways
    similarity_threshold : float
        Threshold for merging similar pathways

    Returns
    -------
    selected_gene_sets : Dict[str, Set[str]]
        Selected representative pathways
    """

    # Filter by size
    filtered_sets = {
        name: genes for name, genes in gene_sets.items()
        if min_size <= len(genes) <= max_size
    }

    if len(filtered_sets) <= max_pathways:
        return filtered_sets

    # Compute similarity matrix
    similarity_matrix, pathway_names = compute_pathway_similarity(filtered_sets)

    # Greedy selection based on coverage and diversity
    selected_pathways = set()
    remaining_pathways = set(pathway_names)
    covered_genes = set()

    while len(selected_pathways) < max_pathways and remaining_pathways:
        # Score each remaining pathway
        scores = {}

        for pathway in remaining_pathways:
            idx = pathway_names.index(pathway)
            pathway_genes = filtered_sets[pathway]

            # Coverage score: new genes covered
            new_genes = pathway_genes - covered_genes
            coverage_score = len(new_genes)

            # Diversity score: average dissimilarity to selected pathways
            if selected_pathways:
                selected_indices = [pathway_names.index(p) for p in selected_pathways]
                similarities = [similarity_matrix[idx, j] for j in selected_indices]
                diversity_score = 1.0 - np.mean(similarities)
            else:
                diversity_score = 1.0

            # Combined score
            scores[pathway] = coverage_score * diversity_score

        # Select pathway with highest score
        best_pathway = max(scores.keys(), key=lambda p: scores[p])
        selected_pathways.add(best_pathway)
        covered_genes.update(filtered_sets[best_pathway])
        remaining_pathways.remove(best_pathway)

    selected_gene_sets = {name: filtered_sets[name] for name in selected_pathways}

    print(f"Selected {len(selected_gene_sets)} representative pathways from {len(gene_sets)} total")

    return selected_gene_sets


__all__ = [
    "compute_pathway_weights",
    "create_pathway_metric",
    "pathway_gene_overlap_matrix",
    "compute_pathway_similarity",
    "select_representative_pathways"
]


