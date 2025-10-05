"""Gate discovery and clustering components."""

from .gmm_kselect import fit_gmm_BICICL
from .wmerge import w2_gaussian, merge_by_w2

__all__ = ["fit_gmm_BICICL", "w2_gaussian", "merge_by_w2"]