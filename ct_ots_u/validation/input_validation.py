"""Input validation and error handling for CT-OTS-U."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..config import CTOTSUConfig

Array = np.ndarray


class CTOTSUError(Exception):
    """Base exception for CT-OTS-U errors."""
    pass


class ValidationError(CTOTSUError):
    """Input validation error."""
    pass


class ConfigurationError(CTOTSUError):
    """Configuration error."""
    pass


class ComputationError(CTOTSUError):
    """Computation error."""
    pass


def validate_array(
    arr: Array,
    name: str,
    min_shape: Optional[tuple] = None,
    max_shape: Optional[tuple] = None,
    dtype: Optional[type] = None,
    finite: bool = True,
    positive: bool = False
) -> None:
    """Validate array properties.

    Parameters
    ----------
    arr : Array
        Array to validate
    name : str
        Name for error messages
    min_shape : tuple, optional
        Minimum shape requirements
    max_shape : tuple, optional
        Maximum shape requirements
    dtype : type, optional
        Required dtype
    finite : bool
        Whether all values must be finite
    positive : bool
        Whether all values must be positive
    """

    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)
        except Exception as e:
            raise ValidationError(f"{name} cannot be converted to numpy array: {e}")

    # Shape validation
    if min_shape is not None:
        if len(arr.shape) != len(min_shape):
            raise ValidationError(
                f"{name} must have {len(min_shape)} dimensions, got {len(arr.shape)}"
            )
        for i, (actual, minimum) in enumerate(zip(arr.shape, min_shape)):
            if minimum is not None and actual < minimum:
                raise ValidationError(
                    f"{name} dimension {i} too small: {actual} < {minimum}"
                )

    if max_shape is not None:
        if len(arr.shape) != len(max_shape):
            raise ValidationError(
                f"{name} must have {len(max_shape)} dimensions, got {len(arr.shape)}"
            )
        for i, (actual, maximum) in enumerate(zip(arr.shape, max_shape)):
            if maximum is not None and actual > maximum:
                raise ValidationError(
                    f"{name} dimension {i} too large: {actual} > {maximum}"
                )

    # Dtype validation
    if dtype is not None and not np.issubdtype(arr.dtype, dtype):
        warnings.warn(f"{name} dtype {arr.dtype} is not {dtype}, converting")

    # Value validation
    if finite and not np.all(np.isfinite(arr)):
        n_nan = np.sum(np.isnan(arr))
        n_inf = np.sum(np.isinf(arr))
        raise ValidationError(
            f"{name} contains non-finite values: {n_nan} NaN, {n_inf} infinite"
        )

    if positive and np.any(arr < 0):
        n_negative = np.sum(arr < 0)
        raise ValidationError(f"{name} contains {n_negative} negative values")


def validate_training_inputs(
    Xs: Array,
    Xt: Array,
    tau: float,
    *,
    rank: Optional[int] = None,
    steps: int = 200,
    lr: float = 1e-2,
    alpha: float = 1e-3,
    min_samples: int = 10,
) -> None:
    """Validate inputs for branch generator training."""

    if min_samples <= 0:
        raise ValidationError(f"min_samples must be positive, got {min_samples}")

    if Xs.size == 0 or Xt.size == 0:
        raise ValidationError("Empty arrays provided")

    if not np.all(np.isfinite(Xs)) or not np.all(np.isfinite(Xt)):
        raise ValidationError("Non-finite values detected in inputs")

    validate_array(Xs, "Xs", min_shape=(1, 2), finite=True)
    validate_array(Xt, "Xt", min_shape=(1, 2), finite=True)

    if Xs.shape[1] != Xt.shape[1]:
        raise ValidationError(
            f"Dimension mismatch: Xs has {Xs.shape[1]} features, Xt has {Xt.shape[1]} features"
        )

    if Xs.shape[0] < min_samples or Xt.shape[0] < min_samples:
        raise ValidationError(
            f"Insufficient samples for training: Xs={Xs.shape[0]}, Xt={Xt.shape[0]}, minimum={min_samples}"
        )

    if tau <= 0:
        raise ValidationError(f"tau must be positive, got {tau}")
    if rank is not None and rank <= 0:
        raise ValidationError(f"rank must be positive, got {rank}")
    if steps <= 0:
        raise ValidationError(f"steps must be positive, got {steps}")
    if lr <= 0:
        raise ValidationError(f"lr must be positive, got {lr}")
    if alpha < 0:
        raise ValidationError(f"alpha must be non-negative, got {alpha}")

    if Xs.shape[0] < 50:
        warnings.warn(f"Very few source samples: {Xs.shape[0]}")
    if Xt.shape[0] < 50:
        warnings.warn(f"Very few target samples: {Xt.shape[0]}")


def validate_positive_param(value: float, name: str) -> None:
    """Ensure parameter is strictly positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive")


def validate_range_param(value: float, name: str, low: float, high: float) -> None:
    """Validate parameter lies within [low, high]."""
    if value < low or value > high:
        raise ValidationError(f"{name} must be in range [{low}, {high}]")


def validate_integer_param(value: Any, name: str) -> None:
    """Ensure parameter is an integer >= 1."""
    if not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be an integer")
    if value <= 0:
        raise ValidationError(f"{name} must be a positive integer")


def validate_training_config(config: Dict[str, Any]) -> None:
    """Validate minimal training configuration dictionary."""
    if not isinstance(config, dict):
        raise ValidationError("config must be a dictionary")

    required = {"tau", "rank", "steps", "lr", "alpha", "reg", "reg_m"}
    missing = [key for key in required if key not in config]
    if missing:
        raise ValidationError(f"Missing required config keys: {missing}")

    validate_positive_param(float(config["tau"]), "tau")
    validate_integer_param(config["rank"], "rank")
    validate_integer_param(config["steps"], "steps")
    validate_positive_param(float(config["lr"]), "lr")
    validate_positive_param(float(config["alpha"]), "alpha")
    validate_positive_param(float(config["reg"]), "reg")
    validate_positive_param(float(config["reg_m"]), "reg_m")

    if "seed" in config:
        validate_integer_param(config["seed"], "seed")


def validate_config(config: Union[CTOTSUConfig, Dict[str, Any]]) -> CTOTSUConfig:
    """Normalize user-provided config into CTOTSUConfig."""
    if isinstance(config, CTOTSUConfig):
        return config
    if not isinstance(config, dict):
        raise ValidationError("config must be CTOTSUConfig or dict")
    validate_training_config(config)
    cfg = CTOTSUConfig()
    cfg.uot.tau = float(config["tau"])
    cfg.uot.eps = float(config.get("reg", cfg.uot.eps))
    cfg.uot.reg_m = float(config.get("reg_m", cfg.uot.reg_m))
    cfg.train.lr = float(config.get("lr", cfg.train.lr))
    cfg.train.steps = int(config.get("steps", cfg.train.steps))
    cfg.train.alpha = float(config.get("alpha", cfg.train.alpha))
    cfg.rank.initial_rank = int(config.get("rank", cfg.rank.initial_rank))
    return cfg

def validate_gene_sets(
    gene_sets: Dict[str, set],
    gene_names: List[str],
    min_size: int = 5,
    max_size: int = 1000
) -> Dict[str, set]:
    """Validate and filter gene sets.

    Parameters
    ----------
    gene_sets : Dict[str, set]
        Gene sets to validate
    gene_names : List[str]
        Available gene names
    min_size : int
        Minimum gene set size
    max_size : int
        Maximum gene set size

    Returns
    -------
    filtered_gene_sets : Dict[str, set]
        Validated and filtered gene sets
    """

    if not isinstance(gene_sets, dict):
        raise ValidationError("gene_sets must be a dictionary")

    if not isinstance(gene_names, (list, tuple, set)):
        raise ValidationError("gene_names must be a list, tuple, or set")

    gene_name_set = set(gene_names)
    filtered_gene_sets = {}

    for pathway_name, pathway_genes in gene_sets.items():
        # Validate pathway name
        if not isinstance(pathway_name, str) or not pathway_name.strip():
            warnings.warn(f"Skipping invalid pathway name: {pathway_name}")
            continue

        # Validate pathway genes
        if not isinstance(pathway_genes, (set, list, tuple)):
            warnings.warn(f"Skipping pathway {pathway_name}: genes not a set/list")
            continue

        pathway_genes = set(pathway_genes)

        # Filter to available genes
        available_genes = pathway_genes & gene_name_set
        if len(available_genes) == 0:
            warnings.warn(f"Skipping pathway {pathway_name}: no genes available")
            continue

        # Size filtering
        if len(available_genes) < min_size:
            warnings.warn(f"Skipping pathway {pathway_name}: too small ({len(available_genes)})")
            continue

        if len(available_genes) > max_size:
            warnings.warn(f"Trimming pathway {pathway_name}: too large ({len(available_genes)})")
            # Keep most common genes (could be improved with better ranking)
            available_genes = set(list(available_genes)[:max_size])

        filtered_gene_sets[pathway_name] = available_genes

    if len(filtered_gene_sets) == 0:
        warnings.warn("No valid gene sets after filtering")

    return filtered_gene_sets


def check_numerical_stability(
    matrix: Array,
    name: str = "matrix",
    max_condition: float = 1e6,
    max_norm: float = 1e6,
) -> tuple[bool, float]:
    """Assess numerical stability of a matrix.

    Returns
    -------
    is_stable : bool
        Whether the matrix satisfies condition and norm thresholds.
    condition_number : float
        Estimated condition number (``inf`` if singular).
    """

    validate_array(matrix, name, finite=True)

    if matrix.ndim != 2:
        raise ValidationError(f"{name} must be 2D matrix, got shape {matrix.shape}")

    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        condition = float("inf")

    det = float(np.linalg.det(matrix))
    if np.isnan(det) or np.isclose(det, 0.0, atol=1e-12):
        condition = float("inf")

    if not np.isfinite(condition):
        condition = float("inf")
        is_condition_ok = False
    else:
        is_condition_ok = condition <= max_condition

    norm = float(np.linalg.norm(matrix))
    if norm > max_norm:
        warnings.warn(f"{name} has large norm: {norm:.2e}")

    is_stable = is_condition_ok and norm <= max_norm
    return is_stable, condition


def validate_matrices(
    matrices: Dict[str, Array],
    *,
    require_square: bool = False,
) -> None:
    """Validate a mapping of named matrices."""

    if not isinstance(matrices, dict):
        raise ValidationError('matrices must be a mapping of name -> matrix')

    for name, matrix in matrices.items():
        if not isinstance(name, str) or not name:
            raise ValidationError('Matrix names must be non-empty strings')
        if matrix is None:
            raise ValidationError(f'Matrix {name} is None')

        try:
            arr = np.asarray(matrix)
        except Exception as exc:
            raise ValidationError(f'Matrix {name} cannot be converted to numpy array: {exc}') from exc

        if arr.ndim != 2:
            raise ValidationError(f'Matrix {name} must be 2D')
        if arr.size == 0:
            raise ValidationError(f'Matrix {name} is empty')
        if require_square and arr.shape[0] != arr.shape[1]:
            raise ValidationError(f'Matrix {name} must be square')
        if not np.all(np.isfinite(arr)):
            raise ValidationError(f'Matrix {name} contains non-finite values')


def validate_optimization_result(
    result: Array,
    loss: float,
    name: str = "optimization result"
) -> None:
    """Validate optimization result.

    Parameters
    ----------
    result : Array
        Optimization result (e.g., generator matrix)
    loss : float
        Final loss value
    name : str
        Result name for error messages
    """

    validate_array(result, name, finite=True)

    # Check loss
    if not np.isfinite(loss):
        raise ComputationError(f"Invalid loss value: {loss}")

    if loss < 0:
        warnings.warn(f"Negative loss may indicate numerical issues: {loss}")

    if loss > 1e6:
        warnings.warn(f"Very large loss may indicate convergence issues: {loss:.2e}")

    # Check result magnitude
    result_norm = np.linalg.norm(result)
    if result_norm == 0:
        warnings.warn(f"{name} is zero matrix")
    elif result_norm > 1e3:
        warnings.warn(f"{name} has large norm: {result_norm:.2e}")


__all__ = [
    "CTOTSUError",
    "ValidationError",
    "ConfigurationError",
    "ComputationError",
    "validate_array",
    "validate_training_inputs",
    "validate_config",
    "validate_gene_sets",
    "check_numerical_stability",
    "validate_matrices",
    "validate_optimization_result"
]