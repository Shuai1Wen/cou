"""Parallel branch training with memory management."""

from __future__ import annotations

import gc
import os
import psutil
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from ..model.train import fit_branch_generator

Array = np.ndarray


class MemoryManager:
    """Memory usage monitoring and management."""

    def __init__(self, max_memory_gb: float = None):
        """Initialize memory manager.

        Parameters
        ----------
        max_memory_gb : float, optional
            Maximum memory usage in GB. If None, uses 80% of available memory.
        """
        if max_memory_gb is None:
            total_memory = psutil.virtual_memory().total / (1024**3)
            self.max_memory_gb = total_memory * 0.8
        else:
            self.max_memory_gb = max_memory_gb

    def get_current_usage(self) -> float:
        """Get current memory usage in GB."""
        return psutil.Process().memory_info().rss / (1024**3)

    def check_memory_available(self, required_gb: float) -> bool:
        """Check if required memory is available."""
        current = self.get_current_usage()
        return (current + required_gb) <= self.max_memory_gb

    def estimate_branch_memory(self, Xs: Array, Xt: Array, rank: int) -> float:
        """Estimate memory required for branch training in GB."""
        n_src, n_tgt, d = Xs.shape[0], Xt.shape[0], Xs.shape[1]

        # Base data
        base_memory = (Xs.nbytes + Xt.nbytes) / (1024**3)

        # Generator matrices and gradients
        generator_memory = 2 * (d * rank * 4) / (1024**3)  # U, V in float32

        # Intermediate computations (matrix exponential, cost matrices)
        intermediate_memory = max(
            (d * d * 4) / (1024**3),  # Matrix exponential
            (n_src * n_tgt * 4) / (1024**3)  # Cost matrix
        )

        # Safety factor
        total = (base_memory + generator_memory + intermediate_memory) * 1.5
        return total

    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        gc.collect()
        if hasattr(gc, 'set_debug'):
            gc.set_debug(0)


def _train_single_branch(args_tuple: Tuple) -> Tuple[int, Optional[Array], float, Dict[str, Any]]:
    """Train a single branch generator (for parallel execution)."""

    branch_id, Xs, Xt_branch, config = args_tuple

    try:
        # Input validation
        if len(Xt_branch) < config.get('min_branch_samples', 10):
            return branch_id, None, float('inf'), {
                'status': 'skipped',
                'reason': f'Too few samples: {len(Xt_branch)}'
            }

        # Memory check
        memory_manager = MemoryManager()
        required_memory = memory_manager.estimate_branch_memory(
            Xs, Xt_branch, config.get('rank', 32)
        )

        if not memory_manager.check_memory_available(required_memory):
            return branch_id, None, float('inf'), {
                'status': 'failed',
                'reason': f'Insufficient memory: {required_memory:.2f}GB required'
            }

        # Train generator
        L_branch = fit_branch_generator(
            Xs, Xt_branch,
            tau=config.get('tau', 0.5),
            rank=config.get('rank', 32),
            steps=config.get('steps', 200),
            lr=config.get('lr', 1e-2),
            alpha=config.get('alpha', 1e-3),
            seed=config.get('seed', 0) + branch_id,
            use_torch=config.get('use_torch', None),
            device=config.get('device', 'cpu')
        )

        # Compute final loss
        from ..model.semigroup import pushforward
        from ..ot.uot_losses import uot_sinkhorn_cost

        Y_pred = pushforward(Xs, L_branch, tau=config.get('tau', 0.5))
        loss, _, _ = uot_sinkhorn_cost(
            Y_pred, Xt_branch,
            reg=config.get('reg', 0.08),
            reg_m=config.get('reg_m', 1.0)
        )

        # Cleanup
        memory_manager.force_cleanup()

        return branch_id, L_branch, float(loss), {
            'status': 'success',
            'n_samples': len(Xt_branch),
            'memory_used_gb': memory_manager.get_current_usage()
        }

    except Exception as e:
        return branch_id, None, float('inf'), {
            'status': 'error',
            'error': str(e)
        }


def parallel_branch_training(
    Xs: Array,
    Xt_branches: List[Array],
    config: Dict[str, Any],
    n_workers: Optional[int] = None,
    use_processes: bool = True,
    verbose: bool = True
) -> Tuple[List[Optional[Array]], List[float], Dict[str, Any]]:
    """Train branch generators in parallel.

    Parameters
    ----------
    Xs : Array
        Source points shared across all branches
    Xt_branches : List[Array]
        List of target points for each branch
    config : Dict
        Training configuration
    n_workers : int, optional
        Number of parallel workers. If None, uses CPU count.
    use_processes : bool
        Whether to use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
    verbose : bool
        Verbose output

    Returns
    -------
    generators : List[Optional[Array]]
        List of trained generators (None for failed branches)
    losses : List[float]
        Training losses for each branch
    metadata : Dict
        Training metadata and statistics
    """

    if n_workers is None:
        n_workers = min(cpu_count(), len(Xt_branches))

    if verbose:
        print(f"Starting parallel training with {n_workers} workers")
        print(f"Training {len(Xt_branches)} branches")

    # Prepare arguments for parallel execution
    args_list = []
    for branch_id, Xt_branch in enumerate(Xt_branches):
        args_list.append((branch_id, Xs, Xt_branch, config))

    # Memory manager for monitoring
    memory_manager = MemoryManager()
    initial_memory = memory_manager.get_current_usage()

    # Choose executor type
    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    generators = [None] * len(Xt_branches)
    losses = [float('inf')] * len(Xt_branches)
    branch_metadata = {}

    try:
        with ExecutorClass(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_branch = {
                executor.submit(_train_single_branch, args): args[0]
                for args in args_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_branch):
                branch_id = future_to_branch[future]

                try:
                    branch_id_result, generator, loss, metadata = future.result()
                    assert branch_id == branch_id_result

                    generators[branch_id] = generator
                    losses[branch_id] = loss
                    branch_metadata[branch_id] = metadata

                    if verbose and metadata['status'] == 'success':
                        print(f"Branch {branch_id} completed: loss={loss:.6f}")
                    elif verbose and metadata['status'] != 'success':
                        print(f"Branch {branch_id} {metadata['status']}: {metadata.get('reason', metadata.get('error', ''))}")

                except Exception as e:
                    if verbose:
                        print(f"Branch {branch_id} failed with exception: {e}")
                    branch_metadata[branch_id] = {'status': 'exception', 'error': str(e)}

    except KeyboardInterrupt:
        if verbose:
            print("Training interrupted by user")
        return generators, losses, {'status': 'interrupted'}

    # Compile final metadata
    final_memory = memory_manager.get_current_usage()
    successful_branches = sum(1 for g in generators if g is not None)

    final_metadata = {
        'n_branches_total': len(Xt_branches),
        'n_branches_successful': successful_branches,
        'n_branches_failed': len(Xt_branches) - successful_branches,
        'success_rate': successful_branches / len(Xt_branches),
        'memory_initial_gb': initial_memory,
        'memory_final_gb': final_memory,
        'memory_increase_gb': final_memory - initial_memory,
        'n_workers': n_workers,
        'executor_type': 'processes' if use_processes else 'threads',
        'branch_metadata': branch_metadata
    }

    if verbose:
        print(f"Parallel training completed:")
        print(f"  Success rate: {successful_branches}/{len(Xt_branches)} ({final_metadata['success_rate']:.1%})")
        print(f"  Memory usage: {initial_memory:.2f}GB → {final_memory:.2f}GB")

    return generators, losses, final_metadata


def adaptive_batch_training(
    Xs: Array,
    Xt_branches: List[Array],
    config: Dict[str, Any],
    max_memory_gb: float = 8.0,
    verbose: bool = True
) -> Tuple[List[Optional[Array]], List[float], Dict[str, Any]]:
    """Train branches in adaptive batches based on memory constraints.

    Parameters
    ----------
    Xs : Array
        Source points
    Xt_branches : List[Array]
        Target branches
    config : Dict
        Training configuration
    max_memory_gb : float
        Maximum memory usage in GB
    verbose : bool
        Verbose output

    Returns
    -------
    generators : List[Optional[Array]]
        Trained generators
    losses : List[float]
        Training losses
    metadata : Dict
        Training metadata
    """

    memory_manager = MemoryManager(max_memory_gb)

    # Estimate memory per branch
    branch_memories = []
    for Xt_branch in Xt_branches:
        mem_est = memory_manager.estimate_branch_memory(
            Xs, Xt_branch, config.get('rank', 32)
        )
        branch_memories.append(mem_est)

    # Sort branches by memory requirement (smallest first)
    sorted_indices = sorted(range(len(Xt_branches)), key=lambda i: branch_memories[i])

    # Adaptive batching
    generators = [None] * len(Xt_branches)
    losses = [float('inf')] * len(Xt_branches)
    all_metadata = {}

    batch_start = 0
    while batch_start < len(sorted_indices):
        # Determine batch size based on memory
        current_memory = memory_manager.get_current_usage()
        available_memory = max_memory_gb - current_memory

        batch_size = 0
        batch_memory = 0.0

        for i in range(batch_start, len(sorted_indices)):
            idx = sorted_indices[i]
            if batch_memory + branch_memories[idx] <= available_memory:
                batch_memory += branch_memories[idx]
                batch_size += 1
            else:
                break

        if batch_size == 0:
            batch_size = 1  # Force at least one branch
            if verbose:
                warnings.warn(f"Memory constraint too tight, forcing batch size 1")

        # Create batch
        batch_indices = sorted_indices[batch_start:batch_start + batch_size]
        batch_branches = [Xt_branches[i] for i in batch_indices]

        if verbose:
            print(f"Processing batch {batch_start//batch_size + 1}: "
                  f"{batch_size} branches, {batch_memory:.2f}GB estimated")

        # Train batch
        batch_generators, batch_losses, batch_metadata = parallel_branch_training(
            Xs, batch_branches, config,
            n_workers=min(batch_size, cpu_count()),
            verbose=verbose
        )

        # Store results
        for i, (gen, loss) in enumerate(zip(batch_generators, batch_losses)):
            original_idx = batch_indices[i]
            generators[original_idx] = gen
            losses[original_idx] = loss

        all_metadata[f'batch_{batch_start//batch_size}'] = batch_metadata

        # Cleanup after batch
        memory_manager.force_cleanup()
        batch_start += batch_size

    # Final metadata
    final_metadata = {
        'mode': 'adaptive_batching',
        'max_memory_gb': max_memory_gb,
        'n_branches': len(Xt_branches),
        'n_successful': sum(1 for g in generators if g is not None),
        'batches': all_metadata
    }

    return generators, losses, final_metadata


__all__ = [
    "parallel_branch_training",
    "adaptive_batch_training",
    "MemoryManager"
]