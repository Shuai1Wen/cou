"""CT-OTS-U utility functions."""
from pathlib import Path

from .seed import set_global_seed, get_rng, reproducible_split
from .metrics import energy_distance, maximum_mean_discrepancy, compute_cost_matrix
from .anndata_io import read_h5ad, write_h5ad, validate_adata_for_ct_ots

__all__ = [
    "set_global_seed",
    "get_rng",
    "reproducible_split",
    "energy_distance",
    "maximum_mean_discrepancy",
    "compute_cost_matrix",
    "read_h5ad",
    "write_h5ad",
    "validate_adata_for_ct_ots",
    "ensure_dir"
]

def ensure_dir(path: Path | str) -> Path:
    """Create directory if missing and return Path"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
