"""AnnData I/O utilities for CT-OTS-U."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False
    ad = None

Array = np.ndarray


def read_h5ad(file_path: str | Path, backed: bool = False) -> "ad.AnnData":
    """Read AnnData from h5ad file with error handling.

    Parameters
    ----------
    file_path : str or Path
        Path to h5ad file
    backed : bool
        Whether to load in backed mode

    Returns
    -------
    adata : AnnData
        Loaded AnnData object
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata is required for read_h5ad")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        adata = ad.read_h5ad(file_path, backed=backed)
        return adata
    except Exception as e:
        raise ValueError(f"Failed to read {file_path}: {e}")


def write_h5ad(
    adata: "ad.AnnData",
    file_path: str | Path,
    compression: Optional[str] = "gzip"
) -> None:
    """Write AnnData to h5ad file.

    Parameters
    ----------
    adata : AnnData
        AnnData object to write
    file_path : str or Path
        Output file path
    compression : str, optional
        Compression method
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata is required for write_h5ad")

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    adata.write_h5ad(file_path, compression=compression)


def validate_adata_for_ct_ots(adata: "ad.AnnData") -> dict:
    """Validate AnnData object for CT-OTS-U compatibility.

    Parameters
    ----------
    adata : AnnData
        AnnData object to validate

    Returns
    -------
    report : dict
        Validation report with issues and recommendations
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata is required for validation")

    issues = []
    recommendations = []

    # Check basic structure
    if adata.n_obs == 0:
        issues.append("No cells (observations) found")
    if adata.n_vars == 0:
        issues.append("No genes (variables) found")

    # Check required observation fields
    required_obs = ["donor_id", "sex", "age", "disease_state"]
    missing_obs = [col for col in required_obs if col not in adata.obs.columns]
    if missing_obs:
        issues.append(f"Missing required obs columns: {missing_obs}")
        recommendations.append("Add required metadata columns for donor stratification")

    # Check for PCA
    if "X_pca" not in adata.obsm:
        recommendations.append("Run PCA preprocessing before CT-OTS-U training")

    # Check for highly variable genes
    if "highly_variable" not in adata.var.columns:
        recommendations.append("Identify highly variable genes for feature selection")

    # Check data sparsity
    if hasattr(adata.X, "toarray"):  # Sparse matrix
        sparsity = 1.0 - (adata.X.nnz / (adata.n_obs * adata.n_vars))
        if sparsity > 0.95:
            warnings.warn(f"Data is very sparse ({sparsity:.1%} zeros)")

    # Check for batch effects
    if "batch" in adata.obs.columns:
        n_batches = adata.obs["batch"].nunique()
        if n_batches > 1:
            recommendations.append(
                f"Consider batch correction for {n_batches} batches"
            )

    # Sample size recommendations
    min_cells_per_condition = 1000
    if "disease_state" in adata.obs.columns:
        condition_counts = adata.obs["disease_state"].value_counts()
        small_conditions = condition_counts[condition_counts < min_cells_per_condition]
        if not small_conditions.empty:
            recommendations.append(
                f"Some conditions have <{min_cells_per_condition} cells: {dict(small_conditions)}"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "recommendations": recommendations,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
        "obsm_keys": list(adata.obsm.keys()),
    }


def extract_matrix(
    adata: "ad.AnnData",
    layer: Optional[str] = None,
    genes: Optional[Sequence[str]] = None
) -> Array:
    """Extract data matrix from AnnData.

    Parameters
    ----------
    adata : AnnData
        AnnData object
    layer : str, optional
        Layer to extract (if None, uses .X)
    genes : Sequence[str], optional
        Gene subset to extract

    Returns
    -------
    X : Array
        Extracted data matrix
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata is required for extract_matrix")

    # Select layer
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]

    # Convert sparse to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Gene subset
    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        if gene_mask.sum() == 0:
            raise ValueError(f"None of the specified genes found in adata.var_names")
        X = X[:, gene_mask]

    return np.asarray(X, dtype=np.float32)


def create_pseudobulk(
    adata: "ad.AnnData",
    groupby: str,
    min_cells: int = 10
) -> "ad.AnnData":
    """Create pseudobulk aggregation by grouping variable.

    Parameters
    ----------
    adata : AnnData
        Single-cell AnnData object
    groupby : str
        Column in adata.obs to group by
    min_cells : int
        Minimum cells required per group

    Returns
    -------
    pseudobulk : AnnData
        Pseudobulk AnnData object
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata is required for create_pseudobulk")

    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")

    # Group cells
    groups = adata.obs.groupby(groupby)
    valid_groups = [name for name, group in groups if len(group) >= min_cells]

    if not valid_groups:
        raise ValueError(f"No groups with at least {min_cells} cells")

    # Aggregate expression
    X_bulk = []
    obs_bulk = []

    for group_name in valid_groups:
        mask = adata.obs[groupby] == group_name
        cells_in_group = adata[mask]

        # Sum expression across cells
        if hasattr(cells_in_group.X, "toarray"):
            X_group = cells_in_group.X.toarray().sum(axis=0)
        else:
            X_group = cells_in_group.X.sum(axis=0)

        X_bulk.append(X_group)

        # Aggregate metadata (take most common values)
        obs_group = cells_in_group.obs.mode().iloc[0].to_dict()
        obs_group[groupby] = group_name
        obs_group["n_cells"] = mask.sum()
        obs_bulk.append(obs_group)

    # Create pseudobulk AnnData
    X_bulk = np.array(X_bulk)
    obs_bulk = pd.DataFrame(obs_bulk)

    pseudobulk = ad.AnnData(
        X=X_bulk,
        obs=obs_bulk,
        var=adata.var.copy()
    )

    return pseudobulk


__all__ = [
    "read_h5ad",
    "write_h5ad",
    "validate_adata_for_ct_ots",
    "extract_matrix",
    "create_pseudobulk"
]