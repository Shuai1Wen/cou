"""Stochastic Weight Averaging (SWA) for generator training."""

from __future__ import annotations

from typing import List, Dict
import numpy as np
import copy


class SWAAccumulator:
    """Accumulate and average generator parameters during training.

    Implements SWA (Stochastic Weight Averaging) to find wider optima
    and improve generalization.

    Reference:
        Izmailov et al. "Averaging Weights Leads to Wider Optima and Better Generalization" (2018)
        https://arxiv.org/abs/1803.05407
    """

    def __init__(
        self,
        start_epoch: int = 80,
        update_freq: int = 1,
        lr_schedule: str = 'constant',  # 'constant' or 'cosine'
        lr_final: float = 0.001,
    ):
        """Initialize SWA accumulator.

        Args:
            start_epoch: Epoch to start SWA averaging
            update_freq: Frequency of SWA updates (in epochs)
            lr_schedule: Learning rate schedule during SWA ('constant' or 'cosine')
            lr_final: Final learning rate for SWA phase
        """
        self.start_epoch = start_epoch
        self.update_freq = update_freq
        self.lr_schedule = lr_schedule
        self.lr_final = lr_final

        self.n_averaged = 0
        self.swa_params: Dict[str, np.ndarray] = {}

    def should_update(self, epoch: int) -> bool:
        """Check if SWA should update at this epoch."""
        if epoch < self.start_epoch:
            return False
        return (epoch - self.start_epoch) % self.update_freq == 0

    def update(self, params: Dict[str, np.ndarray]) -> None:
        """Update SWA averaged parameters.

        Args:
            params: Dictionary of parameter names to arrays (e.g., {'L': matrix, 'b': vector})
        """
        if self.n_averaged == 0:
            # Initialize with first parameters
            self.swa_params = {k: v.copy() for k, v in params.items()}
            self.n_averaged = 1
        else:
            # Running average: swa = (n * swa + new) / (n + 1)
            n = self.n_averaged
            for k, v in params.items():
                if k in self.swa_params:
                    self.swa_params[k] = (n * self.swa_params[k] + v) / (n + 1)
                else:
                    self.swa_params[k] = v.copy()
            self.n_averaged += 1

    def get_averaged_params(self) -> Dict[str, np.ndarray]:
        """Get current SWA averaged parameters."""
        if self.n_averaged == 0:
            return {}
        return {k: v.copy() for k, v in self.swa_params.items()}

    def get_lr(self, epoch: int, lr_init: float) -> float:
        """Get learning rate for SWA phase.

        Args:
            epoch: Current epoch
            lr_init: Initial learning rate

        Returns:
            lr: Learning rate for this epoch
        """
        if epoch < self.start_epoch:
            return lr_init

        if self.lr_schedule == 'constant':
            return self.lr_final
        elif self.lr_schedule == 'cosine':
            # Cosine annealing from lr_init to lr_final
            progress = (epoch - self.start_epoch) / max(1, self.start_epoch)
            progress = min(progress, 1.0)
            return self.lr_final + 0.5 * (lr_init - self.lr_final) * (
                1 + np.cos(np.pi * progress)
            )
        else:
            return self.lr_final


class SWABranchTrainer:
    """Multi-branch trainer with SWA support."""

    def __init__(
        self,
        n_branches: int,
        use_swa: bool = False,
        swa_start_ratio: float = 0.8,
        swa_lr: float = 0.001,
        total_epochs: int = 100,
    ):
        """Initialize SWA branch trainer.

        Args:
            n_branches: Number of branches
            use_swa: Whether to use SWA
            swa_start_ratio: Start SWA at this fraction of total epochs
            swa_lr: Learning rate during SWA
            total_epochs: Total number of training epochs
        """
        self.n_branches = n_branches
        self.use_swa = use_swa
        self.total_epochs = total_epochs

        if use_swa:
            swa_start = int(swa_start_ratio * total_epochs)
            self.swa_accumulators = [
                SWAAccumulator(
                    start_epoch=swa_start,
                    update_freq=1,
                    lr_schedule='constant',
                    lr_final=swa_lr,
                )
                for _ in range(n_branches)
            ]
        else:
            self.swa_accumulators = None

    def should_update_swa(self, branch_idx: int, epoch: int) -> bool:
        """Check if SWA should update for this branch."""
        if not self.use_swa or self.swa_accumulators is None:
            return False
        return self.swa_accumulators[branch_idx].should_update(epoch)

    def update_swa(self, branch_idx: int, params: Dict[str, np.ndarray]) -> None:
        """Update SWA for a branch."""
        if self.use_swa and self.swa_accumulators is not None:
            self.swa_accumulators[branch_idx].update(params)

    def get_swa_params(self, branch_idx: int) -> Dict[str, np.ndarray]:
        """Get SWA averaged parameters for a branch."""
        if not self.use_swa or self.swa_accumulators is None:
            return {}
        return self.swa_accumulators[branch_idx].get_averaged_params()

    def get_swa_lr(self, branch_idx: int, epoch: int, lr_init: float) -> float:
        """Get learning rate for SWA phase."""
        if not self.use_swa or self.swa_accumulators is None:
            return lr_init
        return self.swa_accumulators[branch_idx].get_lr(epoch, lr_init)

    def finalize_all_branches(self) -> List[Dict[str, np.ndarray]]:
        """Get SWA parameters for all branches.

        Returns:
            List of parameter dicts, one per branch
        """
        if not self.use_swa or self.swa_accumulators is None:
            return []

        return [
            acc.get_averaged_params()
            for acc in self.swa_accumulators
        ]


__all__ = [
    'SWAAccumulator',
    'SWABranchTrainer',
]
