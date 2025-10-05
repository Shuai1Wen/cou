"""Random seed management utilities."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

# Try to import torch for GPU seed setting
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set environment variable for child processes
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set PyTorch seeds if available
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Get NumPy random number generator.

    Parameters
    ----------
    seed : int, optional
        Random seed. If None, uses system entropy.

    Returns
    -------
    rng : np.random.Generator
        Random number generator instance
    """
    return np.random.default_rng(seed)


def reproducible_split(
    data_size: int,
    train_frac: float = 0.8,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible train/validation split indices.

    Parameters
    ----------
    data_size : int
        Total number of samples
    train_frac : float
        Fraction of data for training
    seed : int
        Random seed

    Returns
    -------
    train_idx : np.ndarray
        Training indices
    val_idx : np.ndarray
        Validation indices
    """
    rng = get_rng(seed)
    indices = np.arange(data_size)
    rng.shuffle(indices)

    n_train = int(data_size * train_frac)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    return train_idx, val_idx


__all__ = [
    "set_global_seed",
    "get_rng",
    "reproducible_split"
]