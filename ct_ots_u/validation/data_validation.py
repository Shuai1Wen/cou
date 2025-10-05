"""Data quality validation for single-cell data."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .input_validation import ValidationError, CTOTSUError

Array = np.ndarray


def validate_adata(adata, required_obs: Optional[List[str]] = None) -> Dict[str, any]:
    """Validate AnnData object for CT-OTS-U compatibility.

    Parameters
    ----------
    adata : AnnData
        AnnData object to validate
    required_obs : List[str], optional
        Required observation columns

    Returns
    -------
    report : Dict
        Validation report
    """

    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required for data validation")

    if not isinstance(adata, ad.AnnData):
        raise ValidationError("Input must be AnnData object")

    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    # Basic structure checks
    if adata.n_obs == 0:
        report['errors'].append("No cells (observations) found")
        report['valid'] = False

    if adata.n_vars == 0:
        report['errors'].append("No genes (variables) found")
        report['valid'] = False

    # Expression data checks
    if adata.X is None:
        report['errors'].append("No expression data (X) found")
        report['valid'] = False
    else:
        # Convert sparse to dense for checking
        X = adata.X
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = np.asarray(X)

        # Check for non-finite values
        if not np.all(np.isfinite(X_dense)):
            n_nan = np.sum(np.isnan(X_dense))
            n_inf = np.sum(np.isinf(X_dense))
            report['errors'].append(f"Expression data contains {n_nan} NaN and {n_inf} infinite values")
            report['valid'] = False

        # Check for negative values (should be rare in processed data)
        if np.any(X_dense < 0):
            n_negative = np.sum(X_dense < 0)
            report['warnings'].append(f"Expression data contains {n_negative} negative values")

        # Sparsity information
        sparsity = np.mean(X_dense == 0)
        report['info']['sparsity'] = sparsity
        if sparsity > 0.99:
            report['warnings'].append(f"Data is very sparse ({sparsity:.1%} zeros)")

    # Required observation columns
    if required_obs is None:
        required_obs = ['donor_id', 'sex', 'age']

    missing_obs = [col for col in required_obs if col not in adata.obs.columns]
    if missing_obs:
        report['warnings'].append(f"Missing recommended obs columns: {missing_obs}")

    # Check for PCA
    if 'X_pca' not in adata.obsm:
        report['warnings'].append("No PCA found in obsm['X_pca'] - run preprocessing first")

    # Check categorical variables
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object':
            n_unique = adata.obs[col].nunique()
            n_total = len(adata.obs[col])
            if n_unique > n_total * 0.5:
                report['warnings'].append(f"Column {col} has many unique values ({n_unique}/{n_total})")

    # Sample size checks
    if adata.n_obs < 100:
        report['warnings'].append(f"Very few cells: {adata.n_obs}")
    elif adata.n_obs > 50000:
        report['warnings'].append(f"Many cells may be slow to process: {adata.n_obs}")

    if adata.n_vars < 1000:
        report['warnings'].append(f"Few genes: {adata.n_vars}")
    elif adata.n_vars > 30000:
        report['warnings'].append(f"Many genes may be inefficient: {adata.n_vars}")

    # Basic statistics
    report['info'].update({
        'n_obs': adata.n_obs,
        'n_vars': adata.n_vars,
        'obs_columns': list(adata.obs.columns),
        'var_columns': list(adata.var.columns),
        'obsm_keys': list(adata.obsm.keys()),
        'layers_keys': list(adata.layers.keys()) if adata.layers else []
    })

    return report


def check_data_quality(
    adata,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    max_mito_percent: float = 20.0,
    max_ribo_percent: float = 50.0
) -> Dict[str, any]:
    """Perform comprehensive data quality checks.

    Parameters
    ----------
    adata : AnnData
        Single-cell data
    min_genes_per_cell : int
        Minimum genes per cell threshold
    min_cells_per_gene : int
        Minimum cells per gene threshold
    max_mito_percent : float
        Maximum mitochondrial gene percentage
    max_ribo_percent : float
        Maximum ribosomal gene percentage

    Returns
    -------
    qc_report : Dict
        Quality control report
    """

    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for quality control checks")

    qc_report = {
        'passed': True,
        'metrics': {},
        'recommendations': []
    }

    # Make a copy to avoid modifying original
    adata_qc = adata.copy()

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata_qc, percent_top=None, log1p=False, inplace=True)

    # Genes per cell
    genes_per_cell = adata_qc.obs['n_genes_by_counts']
    low_gene_cells = (genes_per_cell < min_genes_per_cell).sum()
    qc_report['metrics']['genes_per_cell'] = {
        'mean': float(genes_per_cell.mean()),
        'median': float(genes_per_cell.median()),
        'low_quality_cells': int(low_gene_cells),
        'percent_low_quality': float(low_gene_cells / len(genes_per_cell) * 100)
    }

    if low_gene_cells > len(genes_per_cell) * 0.1:
        qc_report['recommendations'].append(f"Consider filtering {low_gene_cells} cells with <{min_genes_per_cell} genes")

    # Cells per gene
    cells_per_gene = adata_qc.var['n_cells_by_counts']
    rare_genes = (cells_per_gene < min_cells_per_gene).sum()
    qc_report['metrics']['cells_per_gene'] = {
        'mean': float(cells_per_gene.mean()),
        'median': float(cells_per_gene.median()),
        'rare_genes': int(rare_genes),
        'percent_rare': float(rare_genes / len(cells_per_gene) * 100)
    }

    if rare_genes > len(cells_per_gene) * 0.3:
        qc_report['recommendations'].append(f"Consider filtering {rare_genes} genes expressed in <{min_cells_per_gene} cells")

    # Mitochondrial genes
    mito_genes = [name for name in adata_qc.var_names if name.startswith('MT-')]
    if mito_genes:
        adata_qc.var['mt'] = adata_qc.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata_qc, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        mito_percent = adata_qc.obs['pct_counts_mt']
        high_mito_cells = (mito_percent > max_mito_percent).sum()

        qc_report['metrics']['mitochondrial'] = {
            'n_genes': len(mito_genes),
            'mean_percent': float(mito_percent.mean()),
            'median_percent': float(mito_percent.median()),
            'high_mito_cells': int(high_mito_cells),
            'percent_high_mito': float(high_mito_cells / len(mito_percent) * 100)
        }

        if high_mito_cells > len(mito_percent) * 0.05:
            qc_report['recommendations'].append(f"Consider filtering {high_mito_cells} cells with >{max_mito_percent}% MT genes")
    else:
        qc_report['metrics']['mitochondrial'] = {'n_genes': 0, 'warning': 'No mitochondrial genes found'}

    # Ribosomal genes
    ribo_genes = [name for name in adata_qc.var_names if name.startswith(('RPS', 'RPL'))]
    if ribo_genes:
        adata_qc.var['ribo'] = adata_qc.var_names.str.startswith(('RPS', 'RPL'))
        sc.pp.calculate_qc_metrics(adata_qc, qc_vars=['ribo'], percent_top=None, log1p=False, inplace=True)

        ribo_percent = adata_qc.obs['pct_counts_ribo']
        high_ribo_cells = (ribo_percent > max_ribo_percent).sum()

        qc_report['metrics']['ribosomal'] = {
            'n_genes': len(ribo_genes),
            'mean_percent': float(ribo_percent.mean()),
            'median_percent': float(ribo_percent.median()),
            'high_ribo_cells': int(high_ribo_cells),
            'percent_high_ribo': float(high_ribo_cells / len(ribo_percent) * 100)
        }

        if high_ribo_cells > len(ribo_percent) * 0.1:
            qc_report['recommendations'].append(f"Consider investigating {high_ribo_cells} cells with >{max_ribo_percent}% ribosomal genes")
    else:
        qc_report['metrics']['ribosomal'] = {'n_genes': 0, 'warning': 'No ribosomal genes found'}

    # Total UMI/counts
    total_counts = adata_qc.obs['total_counts']
    qc_report['metrics']['total_counts'] = {
        'mean': float(total_counts.mean()),
        'median': float(total_counts.median()),
        'std': float(total_counts.std()),
        'min': float(total_counts.min()),
        'max': float(total_counts.max())
    }

    # Check for potential doublets (very high counts)
    high_count_threshold = total_counts.quantile(0.99)
    potential_doublets = (total_counts > high_count_threshold * 2).sum()
    if potential_doublets > 0:
        qc_report['recommendations'].append(f"Consider doublet detection for {potential_doublets} cells with very high counts")

    # Gene expression distribution
    if hasattr(adata_qc.X, 'toarray'):
        X = adata_qc.X.toarray()
    else:
        X = adata_qc.X

    mean_expression = np.mean(X, axis=0)
    qc_report['metrics']['gene_expression'] = {
        'mean_expression_mean': float(mean_expression.mean()),
        'mean_expression_std': float(mean_expression.std()),
        'zero_genes': int(np.sum(mean_expression == 0)),
        'highly_expressed_genes': int(np.sum(mean_expression > np.percentile(mean_expression, 99)))
    }

    # Overall assessment
    n_recommendations = len(qc_report['recommendations'])
    if n_recommendations > 3:
        qc_report['passed'] = False
        qc_report['overall'] = 'Significant quality issues detected'
    elif n_recommendations > 0:
        qc_report['overall'] = 'Minor quality issues detected'
    else:
        qc_report['overall'] = 'Data quality looks good'

    return qc_report


def validate_stage_data(
    adata,
    stage_key: str,
    stages: List[str],
    min_cells_per_stage: int = 50
) -> Dict[str, any]:
    """Validate stage-specific data for CT-OTS-U training.

    Parameters
    ----------
    adata : AnnData
        Single-cell data
    stage_key : str
        Column containing stage information
    stages : List[str]
        Required stages
    min_cells_per_stage : int
        Minimum cells required per stage

    Returns
    -------
    stage_report : Dict
        Stage validation report
    """

    stage_report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stage_info': {}
    }

    # Check if stage key exists
    if stage_key not in adata.obs.columns:
        stage_report['errors'].append(f"Stage key '{stage_key}' not found in obs")
        stage_report['valid'] = False
        return stage_report

    # Check stage values
    available_stages = set(adata.obs[stage_key].unique())
    missing_stages = set(stages) - available_stages

    if missing_stages:
        stage_report['errors'].append(f"Missing stages: {missing_stages}")
        stage_report['valid'] = False

    # Count cells per stage
    stage_counts = adata.obs[stage_key].value_counts()

    for stage in stages:
        if stage in stage_counts:
            count = stage_counts[stage]
            stage_report['stage_info'][stage] = {
                'n_cells': int(count),
                'fraction': float(count / len(adata.obs))
            }

            if count < min_cells_per_stage:
                stage_report['warnings'].append(f"Stage '{stage}' has only {count} cells (< {min_cells_per_stage})")
        else:
            stage_report['stage_info'][stage] = {'n_cells': 0, 'fraction': 0.0}

    # Check for extreme imbalances
    if len(stage_counts) > 1:
        max_count = stage_counts.max()
        min_count = stage_counts.min()
        imbalance_ratio = max_count / min_count

        if imbalance_ratio > 10:
            stage_report['warnings'].append(f"Severe stage imbalance: ratio = {imbalance_ratio:.1f}")

    return stage_report


__all__ = [
    "validate_adata",
    "check_data_quality",
    "validate_stage_data"
]