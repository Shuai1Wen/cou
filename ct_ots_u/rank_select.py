# -*- coding: utf-8 -*-
"""Rank selection utilities for CT-OTS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .ct_ots_model import GatedSemigroup
from .ot_metrics import sinkhorn_divergence_uot


@dataclass
class RankStats:
    mean: float
    se: float
    std: float
    n: int
    penalties: List[float]


def _evaluate_rank(
    X0: np.ndarray,
    Xt1: np.ndarray,
    Xt2: np.ndarray,
    Xt12: np.ndarray,
    gating_model,
    rank: int,
    reg: float,
    reg_m: float,
    tau: float,
    top_p: float,
    min_branch: int,
    num_iter_max: int,
    stop_thr: float,
    train_steps: int,
    train_lr: float,
    stable_alpha: float,
    stable_margin: float | None,
    stable_lambda: float,
    soft_weight: float | None,
    stable_use_soft_penalty: bool,
    lambda_soft: float,
    reg_nuc: float,
    use_torch: bool,
    device: str,
    backend: str,
    sinkhorn_backend: str,
    sinkhorn_scaling: float,
    use_torch_compile: bool,
    sinkhorn_minibatch: bool,
    sinkhorn_batch_size: int | None,
) -> Tuple[float, float]:
    model = GatedSemigroup(
        K=gating_model.classes_.shape[0],
        rank=rank,
        reg=reg,
        reg_m=reg_m,
        stable_margin=stable_margin,
        stable_alpha=stable_alpha,
        stable_lambda=stable_lambda,
        stable_soft_weight=soft_weight if soft_weight is not None else 0.5,
        stable_use_soft_penalty=stable_use_soft_penalty,
        stable_lambda_soft=lambda_soft,
        ot_backend=backend,
        sinkhorn_backend=sinkhorn_backend,
        sinkhorn_scaling=sinkhorn_scaling,
        min_branch_samples=min_branch,
        tau=tau,
        num_iter_max=num_iter_max,
        stop_thr=stop_thr,
        train_steps=train_steps,
        train_lr=train_lr,
        reg_nuc=reg_nuc,
        use_swa=False,
        swa_start_ratio=0.8,
        swa_lr=1e-3,
        swa_schedule="constant",
        use_torch=use_torch,
        use_torch_compile=use_torch_compile,
        device=device,
        log_diagnostics=False,
        sinkhorn_minibatch=sinkhorn_minibatch,
        sinkhorn_batch_size=sinkhorn_batch_size,
    ).fit_branchwise(X0, Xt1, gating_model=gating_model, top_p=top_p)
    pred = model.forward_steps(X0, steps=2, delta=1.0)
    err = sinkhorn_divergence_uot(
        pred,
        Xt12,
        reg=reg,
        reg_m=reg_m,
        backend=backend,
        sinkhorn_backend=sinkhorn_backend,
        scaling=sinkhorn_scaling,
        num_iter_max=num_iter_max,
        stop_thr=stop_thr,
    )
    return float(err), float(model.stability_penalty)


def rank_selection_cv(
    X0: np.ndarray,
    Xt1: np.ndarray,
    Xt2: np.ndarray,
    Xt12: np.ndarray,
    gating_model,
    ranks: Sequence[int],
    reg: float,
    reg_m: float,
    tau: float,
    top_p: float,
    repeats: int,
    min_branch: int,
    seed: int = 0,
    num_iter_max: int = 1000,
    stop_thr: float = 1e-6,
    use_one_se: bool = True,
    train_steps: int = 150,
    train_lr: float = 1e-2,
    stable_alpha: float = 1e-3,
    stable_margin: float | None = -1e-3,
    stable_lambda: float = 1.0,
    stable_soft_weight: float | None = None,
    stable_use_soft_penalty: bool = True,
    lambda_soft: float = 0.5,
    reg_nuc: float = 0.0,
    use_torch: bool = False,
    device: str = 'cpu',
    backend: str = 'geomloss',
    sinkhorn_backend: str = 'online',
    sinkhorn_scaling: float = 0.9,
    use_torch_compile: bool = False,
    sinkhorn_minibatch: bool = False,
    sinkhorn_batch_size: int | None = None,
) -> Tuple[int | None, int | None, Dict[int, Dict[str, float]], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rank_scores: Dict[int, List[float]] = {r: [] for r in ranks}
    rank_penalties: Dict[int, List[float]] = {r: [] for r in ranks}
    chosen_per_repeat: List[int] = []

    for _ in range(repeats):
        shuffle_idx = rng.permutation(X0.shape[0])
        X0_rep = X0[shuffle_idx]
        Xt1_rep = Xt1[rng.permutation(Xt1.shape[0])]
        Xt2_rep = Xt2[rng.permutation(Xt2.shape[0])]
        Xt12_rep = Xt12[rng.permutation(Xt12.shape[0])]
        errors_this_rep = {}
        for r in ranks:
            err, penalty = _evaluate_rank(
                X0_rep,
                Xt1_rep,
                Xt2_rep,
                Xt12_rep,
                gating_model,
                rank=r,
                reg=reg,
                reg_m=reg_m,
                tau=tau,
                top_p=top_p,
                min_branch=min_branch,
                num_iter_max=num_iter_max,
                stop_thr=stop_thr,
                train_steps=train_steps,
                train_lr=train_lr,
                stable_alpha=stable_alpha,
                stable_margin=stable_margin,
                stable_lambda=stable_lambda,
                soft_weight=stable_soft_weight,
                stable_use_soft_penalty=stable_use_soft_penalty,
                lambda_soft=lambda_soft,
                reg_nuc=reg_nuc,
                use_torch=use_torch,
                device=device,
                backend=backend,
                sinkhorn_backend=sinkhorn_backend,
                sinkhorn_scaling=sinkhorn_scaling,
                use_torch_compile=use_torch_compile,
                sinkhorn_minibatch=sinkhorn_minibatch,
                sinkhorn_batch_size=sinkhorn_batch_size,
            )
            rank_scores[r].append(err)
            rank_penalties[r].append(penalty)
            errors_this_rep[r] = err
        best_r_rep = min(errors_this_rep, key=errors_this_rep.get)
        chosen_per_repeat.append(best_r_rep)

    stats: Dict[int, Dict[str, float]] = {}
    best_mean = None
    best_r = None
    for r in ranks:
        scores = np.array(rank_scores[r], dtype=float)
        if scores.size == 0:
            continue
        mean = float(scores.mean())
        std = float(scores.std(ddof=1)) if scores.size > 1 else 0.0
        se = float(std / np.sqrt(scores.size)) if scores.size > 1 else 0.0
        stats[r] = {
            'mean': mean,
            'std': std,
            'se': se,
            'n': int(scores.size),
            'penalties': [float(p) for p in rank_penalties[r]],
        }
        if best_mean is None or mean < best_mean:
            best_mean = mean
            best_r = r

    if best_r is None:
        return None, None, stats, {'confidence': 'low', 'frequency': 0.0}

    if use_one_se:
        threshold = stats[best_r]['mean'] + stats[best_r]['se']
        candidate_ranks = [r for r in ranks if r in stats and stats[r]['mean'] <= threshold]
        if candidate_ranks:
            r_pick = min(candidate_ranks)
        else:
            r_pick = best_r
    else:
        r_pick = best_r

    freq = float(chosen_per_repeat.count(r_pick) / len(chosen_per_repeat)) if chosen_per_repeat else 0.0
    if freq >= 0.70:
        confidence = 'high'
    elif freq >= 0.50:
        confidence = 'medium'
    else:
        confidence = 'low'
    conf_info = {'confidence': confidence, 'frequency': freq}

    return r_pick, best_r, stats, conf_info


__all__ = ["rank_selection_cv", "RankStats"]
