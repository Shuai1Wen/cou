from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse


_DEFAULT_OBS_VALUES = {
    'disease_state': 'Unknown',
    'diagnosis': 'Unknown',
    'cell_type': 'Unknown',
    'region': 'Unknown',
    'batch': 'Unknown',
    'braak_stage': '0',
}

_REQUIRED_FIELD_ALIASES = {
    'donor_id': ('donor_id',),
    'sex': ('sex',),
    'age': ('age',),
    'disease_state': ('disease_state', 'diagnosis'),
    'braak_stage': ('braak_stage',),
    'diagnosis': ('diagnosis', 'disease_state'),
    'cell_type': ('cell_type', 'celltype', 'celltype_final'),
    'region': ('region', 'brain_region'),
    'batch': ('batch', 'sample_id', 'batch_id'),
    'pct_counts_mt': ('pct_counts_mt', 'pct_mt'),
}


@dataclass
class ConditionSplits:
    X0: np.ndarray
    Xt1: np.ndarray
    Xt2: np.ndarray
    Xt12: np.ndarray


# ---------------------------------------------------------------------------
# Observation utilities
# ---------------------------------------------------------------------------


def _ensure_column(adata: ad.AnnData, target: str, aliases: Sequence[str]) -> None:
    for alias in aliases:
        if alias in adata.obs.columns:
            if alias != target:
                adata.obs[target] = adata.obs[alias]
            return
    raise ValueError(
        f"AnnData is missing required obs column '{target}'. Available columns: {list(adata.obs.columns)}"
    )



def ensure_common_obs(adata: ad.AnnData) -> None:
    """Ensure AnnData contains required observation fields and harmonise aliases."""
    for target, aliases in _REQUIRED_FIELD_ALIASES.items():
        if target not in adata.obs.columns:
            try:
                _ensure_column(adata, target, aliases)
            except ValueError:
                if target == 'pct_counts_mt':
                    continue
                if target in _DEFAULT_OBS_VALUES:
                    adata.obs[target] = _DEFAULT_OBS_VALUES[target]
                else:
                    raise
    if 'pct_counts_mt' not in adata.obs.columns:
        mt_mask = adata.var_names.str.upper().str.startswith('MT-')
        if mt_mask.sum() == 0:
            adata.obs['pct_counts_mt'] = 0.0
        else:
            X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            total = np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1.0)
            mt_total = np.asarray(X[:, mt_mask].sum(axis=1)).ravel()
            adata.obs['pct_counts_mt'] = mt_total / total * 100.0

    for col in ('donor_id', 'sex', 'disease_state', 'diagnosis', 'cell_type', 'region', 'batch'):
        adata.obs[col] = adata.obs[col].astype(str)

    adata.obs['age'] = pd.to_numeric(adata.obs['age'], errors='coerce')
    adata.obs['pct_counts_mt'] = pd.to_numeric(adata.obs['pct_counts_mt'], errors='coerce').fillna(0.0)


# ---------------------------------------------------------------------------
# Helpers for preprocessing pipelines
# ---------------------------------------------------------------------------


def _candidate_count_layers(adata: ad.AnnData) -> Sequence[str]:
    return ('counts', 'raw_counts', 'raw')


def _get_counts_matrix(adata: ad.AnnData):
    for key in _candidate_count_layers(adata):
        if key in adata.layers:
            return adata.layers[key]
    if adata.raw is not None:
        return adata.raw.X
    return None


def _looks_logged(adata: ad.AnnData, counts_layer) -> bool:
    if counts_layer is not None:
        return False
    if 'log1p' in adata.uns:
        return True
    if 'highly_variable' in adata.var.columns:
        return True
    X = adata.X
    if sparse.issparse(X):
        sample = X[: min(X.shape[0], 256), : min(X.shape[1], 256)].toarray()
    else:
        sample = np.asarray(X[: min(X.shape[0], 256), : min(X.shape[1], 256)])
    if sample.size == 0:
        return False
    max_val = float(np.max(sample))
    if max_val <= 25.0:
        frac_non_int = float(np.mean(np.abs(sample - np.round(sample)) > 1e-3))
        if frac_non_int > 0.1:
            return True
    return False


def _expm1_sparse(matrix):
    if sparse.issparse(matrix):
        approx = matrix.copy().astype(np.float64)
        approx.data = np.expm1(approx.data)
        approx.eliminate_zeros()
        return approx
    return np.expm1(np.asarray(matrix, dtype=np.float64))


def _subset_existing_hvgs(adata: ad.AnnData, n_top: int) -> ad.AnnData:
    if 'highly_variable' not in adata.var.columns:
        return adata
    hv_mask = adata.var['highly_variable'].astype(bool).to_numpy()
    if hv_mask.sum() == 0:
        return adata
    if 'highly_variable_rank' in adata.var.columns:
        ranks = adata.var['highly_variable_rank'].to_numpy(dtype=float)
        valid = np.isfinite(ranks)
        order = np.argsort(ranks[valid])
        selected = np.where(valid)[0][order[: min(n_top, order.size)]]
        mask = np.zeros_like(hv_mask)
        mask[selected] = True
        hv_mask = mask
    else:
        idx = np.where(hv_mask)[0][:n_top]
        mask = np.zeros_like(hv_mask)
        mask[idx] = True
        hv_mask = mask
    subset = adata[:, hv_mask].copy()
    if 'highly_variable' in subset.var.columns:
        subset.var['highly_variable'] = True
    if 'highly_variable_rank' in subset.var.columns:
        subset.var['highly_variable_rank'] = np.arange(subset.n_vars)
    return subset


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------


def preprocess_adata(
    adata: ad.AnnData,
    n_pcs: int = 64,
    seed: int = 0,
    hvgs: int = 3000,
    scale_max: float = 10.0,
    min_counts: int = 500,
    min_genes: int = 200,
    min_cells: int = 10,
    pct_mt_limit: float | None = 12.0,
) -> ad.AnnData:
    """Normalize, QC filter, select HVGs and compute PCA embedding."""

    ensure_common_obs(adata)

    if adata.n_obs == 0:
        raise ValueError('AnnData contains no cells prior to preprocessing')

    if adata.X is None:
        raise ValueError('AnnData.X is empty; raw counts required for preprocessing')

    counts_layer = _get_counts_matrix(adata)
    logged_input = _looks_logged(adata, counts_layer)

    if counts_layer is not None:
        qc_matrix = counts_layer
        approx_counts = False
    elif logged_input:
        qc_matrix = _expm1_sparse(adata.X)
        approx_counts = True
    else:
        qc_matrix = adata.X
        approx_counts = False

    if sparse.issparse(qc_matrix):
        counts = np.asarray(qc_matrix.sum(axis=1)).ravel()
        genes = np.asarray((qc_matrix > 0).sum(axis=1)).ravel()
    else:
        counts = np.asarray(qc_matrix.sum(axis=1)).ravel()
        genes = np.asarray((qc_matrix > 0).sum(axis=1)).ravel()

    adata.obs['n_counts'] = counts
    adata.obs['n_genes'] = genes

    pct_mt_series = pd.to_numeric(adata.obs['pct_counts_mt'], errors='coerce').fillna(0.0)
    adata.obs['pct_counts_mt'] = pct_mt_series

    cell_mask = np.ones(adata.n_obs, dtype=bool)
    if not approx_counts:
        cell_mask &= (adata.obs['n_counts'] >= min_counts) & (adata.obs['n_genes'] >= min_genes)
    if pct_mt_limit is not None:
        cell_mask &= pct_mt_series <= float(pct_mt_limit)

    if cell_mask.sum() == 0:
        raise ValueError('All cells were filtered out by QC thresholds')
    if not cell_mask.all():
        adata = adata[cell_mask].copy()
        if sparse.issparse(qc_matrix):
            qc_matrix = qc_matrix[cell_mask]
        else:
            qc_matrix = qc_matrix[cell_mask]

    if not approx_counts:
        sc.pp.filter_genes(adata, min_cells=min_cells)
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError('No data remaining after QC filtering')
    else:
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError('No data remaining after QC filtering')

    if not logged_input or counts_layer is not None:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(
            adata,
            flavor='seurat_v3',
            n_top_genes=min(hvgs, adata.n_vars),
            subset=True,
        )
    else:
        adata = _subset_existing_hvgs(adata, min(hvgs, adata.n_vars))
        if 'highly_variable' not in adata.var.columns:
            adata.var['highly_variable'] = True

    sc.pp.scale(adata, max_value=scale_max)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack', random_state=seed)
    return adata


# ---------------------------------------------------------------------------
# Misc helpers used downstream
# ---------------------------------------------------------------------------


def subsample_matrix(
    X: np.ndarray,
    n: int,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if X.shape[0] <= n or n <= 0:
        return X
    idx = rng.choice(X.shape[0], n, replace=False)
    return X[idx]


def donor_train_valid_split(
    adata: ad.AnnData,
    train_frac: float = 0.8,
    seed: int = 0,
    donor_key: str = 'donor_id',
) -> Tuple[np.ndarray, np.ndarray]:
    donors = adata.obs[donor_key].astype(str).values
    unique = np.unique(donors)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = max(1, int(len(unique) * train_frac))
    train_donors = set(unique[:n_train])
    train_mask = np.array([d in train_donors for d in donors])
    valid_mask = ~train_mask
    if valid_mask.sum() == 0:
        valid_mask = train_mask.copy()
    return train_mask, valid_mask


def make_pseudotime_bins(
    adata: ad.AnnData,
    cond_key: str = 'diagnosis',
    ctrl_label: str = 'Control',
    n_neighbors: int = 30,
    seed: int = 0,
    q_low: float = 0.33,
    q_high: float = 0.66,
) -> ad.AnnData:
    """Annotate AnnData with early/mid/late bins via DPT pseudotime."""

    adata = adata.copy()
    if 'X_pca' not in adata.obsm:
        raise ValueError('PCA embedding (X_pca) required before DPT bins')

    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=n_neighbors, random_state=seed)
    ctrl_idx = np.where(adata.obs[cond_key].astype(str) == str(ctrl_label))[0]
    if ctrl_idx.size == 0:
        ctrl_idx = np.arange(min(10, adata.n_obs))
    adata.uns['iroot'] = int(ctrl_idx[0])
    sc.tl.dpt(adata)
    values = adata.obs['dpt_pseudotime'].to_numpy()
    lo, hi = np.quantile(values, [q_low, q_high])
    bins = np.full(values.shape, 'mid', dtype=object)
    bins[values <= lo] = 'early'
    bins[values >= hi] = 'late'
    adata.obs['stage_bin'] = bins
    return adata


def extract_condition_arrays(
    adata: ad.AnnData,
    stage_key: str,
    stage_map: Mapping[str, Sequence[str]],
    sample_size: int,
    seed: int = 0,
) -> ConditionSplits:
    """Extract early/mid/late latent matrices from AnnData."""

    if 'X_pca' not in adata.obsm:
        raise ValueError('PCA embedding missing; run preprocess_adata first')

    required_keys = {'early', 'mid', 'late'}
    if set(stage_map.keys()) != required_keys:
        raise ValueError(
            f"Stage map must provide labels for {required_keys}, got {set(stage_map.keys())}"
        )

    rng = np.random.default_rng(seed)

    def _sample(stage: str) -> np.ndarray:
        labels = stage_map[stage]
        mask = adata.obs[stage_key].astype(str).isin(labels)
        X = adata.obsm['X_pca'][mask]
        if X.shape[0] == 0:
            raise ValueError(f"No cells found for stage '{stage}' with labels {labels}")
        if sample_size > 0 and X.shape[0] > sample_size:
            idx = rng.choice(X.shape[0], sample_size, replace=False)
            return X[idx]
        return X

    X0 = _sample('early')
    Xt1 = _sample('mid')
    Xt2 = _sample('late')
    Xt12 = np.vstack([Xt1, Xt2])
    return ConditionSplits(X0=X0, Xt1=Xt1, Xt2=Xt2, Xt12=Xt12)


__all__ = [
    'ConditionSplits',
    'ensure_common_obs',
    'preprocess_adata',
    'subsample_matrix',
    'donor_train_valid_split',
    'make_pseudotime_bins',
    'extract_condition_arrays',
]
