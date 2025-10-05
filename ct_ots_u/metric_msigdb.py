"""MSigDB-based metric helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


def load_gmt(path: Path) -> Dict[str, set[str]]:
    gene_sets: Dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = set(g.upper() for g in parts[2:])
            gene_sets[name] = genes
    return gene_sets


def build_weight_matrix(
    genes: Iterable[str],
    gmt_paths: Iterable[Path],
    mode: str = "diag",
    alpha: float = 1.0,
    default: float = 1.0,
) -> np.ndarray:
    genes = [g.upper() for g in genes]
    weights = np.full(len(genes), default, dtype=float)
    if mode != "diag":
        raise NotImplementedError("Only diagonal weighting supported in current pipeline")

    counts = np.zeros(len(genes), dtype=float)
    for gmt in gmt_paths:
        gene_sets = load_gmt(Path(gmt))
        for s in gene_sets.values():
            mask = np.array([gene in s for gene in genes], dtype=float)
            counts += mask
    counts = counts / counts.max() if counts.max() > 0 else counts
    weights = default + alpha * counts
    return np.diag(weights)


def project_metric_to_latent(W_gene: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project gene-space metric matrix into latent PCA space."""
    return components @ W_gene @ components.T
