"""Unified configuration management for CT-OTS-U algorithm."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

# Environment variable defaults
CT_OTS_TAU_DEFAULT = float(os.environ.get('CT_OTS_TAU', '0.5'))
CT_OTS_REG_DEFAULT = float(os.environ.get('CT_OTS_REG', '0.08'))
CT_OTS_REGM_DEFAULT = float(os.environ.get('CT_OTS_REGM', '1.0'))
CT_OTS_SAMPLE_DEFAULT = int(os.environ.get('CT_OTS_SAMPLE', '2000'))
CT_OTS_SEED_DEFAULT = int(os.environ.get('CT_OTS_SEED', '42'))
CT_OTS_DELTA_TOL = float(os.environ.get('CT_OTS_DELTA_TOL', '0.01'))


@dataclass
class UOTCfg:
    """Unbalanced Optimal Transport configuration."""
    tau: float = CT_OTS_TAU_DEFAULT      # 分段步长 Δt
    eps: float = CT_OTS_REG_DEFAULT      # entropic reg (ε)
    reg_m: float = CT_OTS_REGM_DEFAULT   # UOT 边际松弛
    backend: str = 'geomloss'            # Optimal transport backend
    sinkhorn_backend: str = 'online'     # GeomLoss backend ('online', 'multiscale', etc.)
    sinkhorn_scaling: float = 0.9        # ε-降温比例
    max_iter: int = 1000
    stop_thr: float = 1e-6


@dataclass
class GateCfg:
    """Gate discovery and clustering configuration."""
    mode: str = 'bayesian'               # 'gaussian' or 'bayesian' gating
    k_grid: Tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10)
    min_cluster_frac: float = 0.07       # 最小簇占比
    wmerge_thresh: float = 0.5           # Wasserstein 合并阈值
    top_p: float = 0.7                   # 分支 Top-p 用于一致性检验
    covariance_type: str = 'tied'        # GMM covariance type
    reg_covar: float = 1e-4              # GMM regularization
    weight_concentration_prior: float | None = None  # DP concentration prior
    gmm_max_iter: int = 500              # Maximum EM iterations for mixture
    gmm_n_init: int = 5                  # Number of Gaussian initialisations
    gating_C: float = 1.0                # Logistic regression regularization
    gating_max_iter: int = 300           # Logistic regression max iterations


@dataclass
class RankCfg:
    """Rank selection configuration."""
    initial_rank: int = 16
    grid: Tuple[int, ...] = (16, 24, 32, 40, 48)
    cv_repeats: int = 3
    one_se: bool = True                  # Use 1-SE rule for selection


@dataclass
class StableCfg:
    """Stability projection configuration."""
    alpha: float = 1e-3                 # 谱投影最小衰减
    max_spectral_margin: float = -1e-3  # Strict negative margin for stability
    lambda_: float = 1.0                # Stability penalty weight (hard projection)
    lambda_soft: float = 0.5            # Soft penalty weight (quadratic)
    soft_weight: float = 0.5            # Blend between raw and projected
    log_raw_stability: bool = True      # Record raw/projection stability metrics
    eval_disable_projection: bool = False  # Disable projections during evaluation
    use_soft_penalty: bool = True       # Use quadratic soft penalty during training


@dataclass
class EvalTol:
    """Evaluation tolerances and thresholds."""
    semigroup_improve_min: float = 0.05  # ≥5% improvement required
    delta_tolerance: float = CT_OTS_DELTA_TOL  # |delta| < tol treated as tie
    consistency_alpha: float = 0.05      # Statistical significance level
    branch_rel_thresh: float = 0.05      # Branch improvement threshold


@dataclass
class DataCfg:
    """Data preprocessing configuration."""
    n_pcs: int = 64                      # Number of principal components
    hvg_flavor: str = "seurat_v3"        # Highly variable genes method
    n_hvg: int = 3000                    # Number of highly variable genes
    min_counts: int = 500                # Minimum counts per cell
    min_genes: int = 200                 # Minimum genes per cell
    min_cells: int = 10                  # Minimum cells per gene
    pct_mt_pbmc: float = 12.0            # PBMC mitochondrial percentage limit
    pct_mt_microglia: float = 15.0       # Microglia mitochondrial percentage limit
    target_sum: float = 1e4              # Normalization target
    max_value: float = 10.0              # Scaling max value
    sample_size: int = CT_OTS_SAMPLE_DEFAULT  # Sample size per condition
    train_frac: float = 0.8              # Training fraction for donor splits


@dataclass
class HyperparamCfg:
    """Hyperparameter scanning configuration."""
    tau_grid: Tuple[float, ...] = (0.5, 1.0, 2.0)
    reg_grid: Tuple[float, ...] = (0.03, 0.05, 0.08)
    reg_m_grid: Tuple[float, ...] = (0.5, 1.0, 2.0)
    repeats: int = 3                     # Number of repeats for grid search
    min_cluster_frac_grid: Tuple[float, ...] = (0.05, 0.07, 0.10)
    merge_wass_grid: Tuple[float, ...] = (0.4, 0.5, 0.6)


@dataclass
class TrainCfg:
    """Training configuration."""
    steps: int = 100
    lr: float = 1e-2
    alpha: float = 1e-3
    use_torch: bool = False
    device: str = 'cpu'
    seed: int = CT_OTS_SEED_DEFAULT
    clip_neg_dist_eps: float = 1e-8     # Numerical floor for Sinkhorn distances

    # SWA (Stochastic Weight Averaging) config
    use_swa: bool = False               # Enable SWA
    swa_start_ratio: float = 0.8        # Start SWA at 80% of training
    swa_lr: float = 1e-3                # SWA learning rate
    swa_schedule: str = 'constant'      # 'constant' or 'cosine'

    # Domain adaptation config
    use_dann: bool = False              # Enable domain adversarial gating
    lambda_domain: float = 0.1          # Domain adversarial weight

    # Batch integration config
    use_harmony: bool = False           # Enable Harmony batch correction
    harmony_theta: float = 2.0          # Harmony diversity penalty
    harmony_lambda: float = 1.0         # Harmony ridge regularization

    # UOT plateau selection
    use_plateau_lock: bool = False      # Enable UOT plateau locking
    plateau_prefer_small_tau: bool = True  # Prefer smaller tau in plateau


@dataclass
class CTOTSUConfig:
    """Complete CT-OTS-U algorithm configuration."""
    uot: UOTCfg = field(default_factory=UOTCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    rank: RankCfg = field(default_factory=RankCfg)
    stable: StableCfg = field(default_factory=StableCfg)
    eval_tol: EvalTol = field(default_factory=EvalTol)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    hyperparam: HyperparamCfg = field(default_factory=HyperparamCfg)

    # Global settings
    seed: int = CT_OTS_SEED_DEFAULT
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "CTOTSUConfig":
        """Create configuration from environment variables."""
        return cls()

    def auto_detect_device(self, verbose: bool | None = None) -> None:
        """
        Automatically detect and configure the best available device.

        This method updates self.train.device, self.uot.backend, and
        self.train.use_torch based on hardware availability.

        Args:
            verbose: If True, print detection results. If None, use self.verbose.
        """
        from .utils.device_detect import detect_best_device

        show_info = verbose if verbose is not None else self.verbose
        device, backend, use_torch = detect_best_device(verbose=show_info)

        self.train.device = device
        self.uot.backend = backend
        self.train.use_torch = use_torch

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "uot": {
                "tau": self.uot.tau,
                "eps": self.uot.eps,
                "reg_m": self.uot.reg_m,
                "backend": self.uot.backend,
                "sinkhorn_backend": self.uot.sinkhorn_backend,
                "sinkhorn_scaling": self.uot.sinkhorn_scaling,
                "max_iter": self.uot.max_iter,
                "stop_thr": self.uot.stop_thr,
            },
            "gate": {
                "mode": self.gate.mode,
                "k_grid": self.gate.k_grid,
                "min_cluster_frac": self.gate.min_cluster_frac,
                "wmerge_thresh": self.gate.wmerge_thresh,
                "top_p": self.gate.top_p,
                "covariance_type": self.gate.covariance_type,
                "reg_covar": self.gate.reg_covar,
                "weight_concentration_prior": self.gate.weight_concentration_prior,
                "gmm_max_iter": self.gate.gmm_max_iter,
                "gmm_n_init": self.gate.gmm_n_init,
                "gating_C": self.gate.gating_C,
                "gating_max_iter": self.gate.gating_max_iter,
            },
            "hyperparam": {
                "tau_grid": self.hyperparam.tau_grid,
                "reg_grid": self.hyperparam.reg_grid,
                "reg_m_grid": self.hyperparam.reg_m_grid,
                "repeats": self.hyperparam.repeats,
            },
            "rank": {
                "initial_rank": self.rank.initial_rank,
                "grid": self.rank.grid,
                "cv_repeats": self.rank.cv_repeats,
                "one_se": self.rank.one_se,
            },
            "stable": {
                "alpha": self.stable.alpha,
                "max_spectral_margin": self.stable.max_spectral_margin,
                "lambda_": self.stable.lambda_,
                "lambda_soft": self.stable.lambda_soft,
                "soft_weight": self.stable.soft_weight,
                "log_raw_stability": self.stable.log_raw_stability,
                "eval_disable_projection": self.stable.eval_disable_projection,
                "use_soft_penalty": self.stable.use_soft_penalty,
            },
            "eval_tol": {
                "semigroup_improve_min": self.eval_tol.semigroup_improve_min,
                "delta_tolerance": self.eval_tol.delta_tolerance,
                "consistency_alpha": self.eval_tol.consistency_alpha,
            },
            "data": {
                "n_pcs": self.data.n_pcs,
                "hvg_flavor": self.data.hvg_flavor,
                "n_hvg": self.data.n_hvg,
                "min_counts": self.data.min_counts,
                "min_genes": self.data.min_genes,
                "min_cells": self.data.min_cells,
                "pct_mt_pbmc": self.data.pct_mt_pbmc,
                "pct_mt_microglia": self.data.pct_mt_microglia,
                "sample_size": self.data.sample_size,
                "train_frac": self.data.train_frac,
            },
            "train": {
                "steps": self.train.steps,
                "lr": self.train.lr,
                "alpha": self.train.alpha,
                "use_torch": self.train.use_torch,
                "device": self.train.device,
                "seed": self.train.seed,
                "clip_neg_dist_eps": self.train.clip_neg_dist_eps,
                "use_swa": self.train.use_swa,
                "swa_start_ratio": self.train.swa_start_ratio,
                "swa_lr": self.train.swa_lr,
                "swa_schedule": self.train.swa_schedule,
                "use_dann": self.train.use_dann,
                "lambda_domain": self.train.lambda_domain,
                "use_harmony": self.train.use_harmony,
                "harmony_theta": self.train.harmony_theta,
                "harmony_lambda": self.train.harmony_lambda,
                "use_plateau_lock": self.train.use_plateau_lock,
                "plateau_prefer_small_tau": self.train.plateau_prefer_small_tau,
            },
            "seed": self.seed,
            "verbose": self.verbose,
        }


# Global default configuration instance
default_config = CTOTSUConfig()

__all__ = [
    "UOTCfg",
    "GateCfg",
    "RankCfg",
    "StableCfg",
    "EvalTol",
    "DataCfg",
    "HyperparamCfg",
    "CTOTSUConfig",
    "default_config",
]
