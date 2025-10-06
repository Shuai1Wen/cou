# -*- coding: utf-8 -*-
"""CT-OTS gated semigroup model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import AlignCfg, DynamicsCfg, RegularCfg
from .model.semigroup import pushforward
from .model.train import fit_branch_generator
from .stability import stability_penalty
from .ot_metrics import sinkhorn_divergence_bal


def _argmax_labels(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs).argmax(axis=1)


def _consistency_improvement(direct_err: float, composed_err: float, eps: float) -> float:
    d = max(direct_err, eps)
    c = max(composed_err, eps)
    return 1.0 - (c / d)


@dataclass
class SemigroupReport:
    mid_error: float
    two_step_error: float
    direct_error: float
    consistency_error: float
    improvement: float
    stability_penalty: float


class GatedSemigroup:
    def __init__(
        self,
        K: int,
        *,
        rank: int = 48,
        reg: float = 0.05,
        reg_m: float = 1.0,
        stable_margin: float | None = -1e-3,
        stable_alpha: float = 1e-3,
        stable_lambda: float = 1.0,
        stable_soft_weight: float = 0.5,
        stable_use_soft_penalty: bool = True,
        stable_lambda_soft: float = 0.5,
        ot_backend: str = "geomloss",
        sinkhorn_backend: str = "online",
        sinkhorn_scaling: float = 0.9,
        min_branch_samples: int = 64,
        tau: float = 1.0,
        num_iter_max: int = 1000,
        stop_thr: float = 1e-6,
        train_steps: int = 150,
        train_lr: float = 1e-2,
        use_swa: bool = False,
        swa_start_ratio: float = 0.8,
        swa_lr: float = 1e-3,
        swa_schedule: str = "constant",
        reg_nuc: float = 0.0,
        use_torch: bool = False,
        use_torch_compile: bool = False,
        device: str = "cpu",
        log_diagnostics: bool = True,
        sinkhorn_minibatch: bool = False,
        sinkhorn_batch_size: int | None = None,
        dynamics_cfg: DynamicsCfg | None = None,
        regular_cfg: RegularCfg | None = None,
        align_cfg: AlignCfg | None = None,
    ) -> None:
        self.K = K
        self.rank = rank
        self.reg = reg
        self.reg_m = reg_m
        self.stable_margin = stable_margin
        self.stable_alpha = stable_alpha
        self.stable_lambda = stable_lambda
        self.stable_soft_weight = stable_soft_weight
        self.stable_use_soft_penalty = stable_use_soft_penalty
        self.stable_lambda_soft = stable_lambda_soft
        self.ot_backend = ot_backend
        self.sinkhorn_backend = sinkhorn_backend
        self.sinkhorn_scaling = sinkhorn_scaling
        self.min_branch_samples = min_branch_samples
        self.tau = tau
        self.num_iter_max = num_iter_max
        self.stop_thr = stop_thr
        self.train_steps = train_steps
        self.train_lr = train_lr
        self.use_swa = use_swa
        self.swa_start_ratio = swa_start_ratio
        self.swa_lr = swa_lr
        self.swa_schedule = swa_schedule
        self.reg_nuc = reg_nuc
        self.use_torch = use_torch
        self.use_torch_compile = use_torch_compile
        self.device = device
        self.log_diagnostics = log_diagnostics
        self.sinkhorn_minibatch = sinkhorn_minibatch
        self.sinkhorn_batch_size = sinkhorn_batch_size

        self.dynamics_cfg = dynamics_cfg or DynamicsCfg()
        self.regular_cfg = regular_cfg or RegularCfg(
            lyapunov_alpha=stable_alpha,
            lyapunov_lambda=stable_lambda,
        )
        self.align_cfg = align_cfg or AlignCfg()

        self.gating_model = None
        self.L_list: List[np.ndarray] = []
        self.bias_list: List[np.ndarray] = []
        self.stability_diagnostics: List[dict[str, float]] = []

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray, gating_model) -> "GatedSemigroup":
        self.gating_model = gating_model
        self._fit_maps(X_src, X_tgt)
        return self

    def fit_pair(self, X_src: np.ndarray, X_tgt: np.ndarray) -> "GatedSemigroup":
        if self.gating_model is None:
            raise ValueError("Gating model not set; call fit() first")
        self._fit_maps(X_src, X_tgt)
        return self

    def _top_indices(self, probs: np.ndarray, k: int, top_p: float) -> np.ndarray:
        if top_p <= 0.0:
            return np.arange(probs.shape[0])
        scores = probs[:, k]
        if top_p >= 1.0:
            threshold = np.min(scores) - 1e-8
        else:
            threshold = np.quantile(scores, 1.0 - top_p)
        idx = np.where(scores >= threshold)[0]
        if idx.size == 0:
            idx = np.argsort(scores)[-max(1, int(np.ceil(scores.size * top_p))):]
        return idx

    def _fit_maps(
        self,
        X_src: np.ndarray,
        X_tgt: np.ndarray,
        top_p: float | None = None,
    ) -> None:
        probs_src = self.gating_model.predict_proba(X_src)
        probs_tgt = self.gating_model.predict_proba(X_tgt)
        labels_src = _argmax_labels(probs_src)
        labels_tgt = _argmax_labels(probs_tgt)

        self.L_list = []
        self.bias_list = []
        self.stability_diagnostics = []
        rng = np.random.default_rng(
            self.gating_model.random_state if hasattr(self.gating_model, "random_state") else 0
        )

        for k in range(self.K):
            idx_src = np.where(labels_src == k)[0]
            idx_tgt = np.where(labels_tgt == k)[0]
            if top_p is not None:
                idx_src = self._top_indices(probs_src, k, top_p)
                idx_tgt = self._top_indices(probs_tgt, k, top_p)

            if idx_src.size < self.min_branch_samples or idx_tgt.size < self.min_branch_samples:
                Xs = X_src
                Xt = X_tgt
            else:
                Xs = X_src[idx_src]
                Xt = X_tgt[idx_tgt]

            result = fit_branch_generator(
                Xs,
                Xt,
                tau=self.tau,
                rank=self.rank,
                steps=self.train_steps,
                lr=self.train_lr,
                alpha=self.stable_alpha,
                reg_nuc=self.reg_nuc,
                seed=rng.integers(1 << 31),
                use_torch=self.use_torch,
                use_torch_compile=self.use_torch_compile,
                device=self.device,
                reg=self.reg,
                reg_m=self.reg_m,
                num_iter_max=self.num_iter_max,
                stop_thr=self.stop_thr,
                stable_margin=self.stable_margin,
                soft_weight=self.stable_soft_weight,
                use_soft_penalty=self.stable_use_soft_penalty,
                lambda_soft=self.stable_lambda_soft,
                use_swa=self.use_swa,
                swa_start_ratio=self.swa_start_ratio,
                swa_lr=self.swa_lr,
                swa_schedule=self.swa_schedule,
                log_diagnostics=self.log_diagnostics,
                backend=self.ot_backend,
                sinkhorn_backend=self.sinkhorn_backend,
                sinkhorn_scaling=self.sinkhorn_scaling,
                sinkhorn_minibatch=self.sinkhorn_minibatch,
                sinkhorn_batch_size=self.sinkhorn_batch_size,
                dynamics_cfg=self.dynamics_cfg,
                regular_cfg=self.regular_cfg,
                align_cfg=self.align_cfg,
            )

            if isinstance(result, tuple):
                Lk, diag = result
            else:
                Lk, diag = result, {}

            projected = pushforward(Xs, Lk, tau=self.tau)
            bias = Xt.mean(axis=0) - projected.mean(axis=0)

            self.L_list.append(Lk)
            self.bias_list.append(bias)
            if self.log_diagnostics:
                self.stability_diagnostics.append(diag)
            else:
                self.stability_diagnostics.append({})

    def fit_branchwise(
        self,
        X_src: np.ndarray,
        X_tgt: np.ndarray,
        gating_model,
        top_p: float = 0.7,
    ) -> "GatedSemigroup":
        self.gating_model = gating_model
        self._fit_maps(X_src, X_tgt, top_p=top_p)
        return self

    def forward(self, X: np.ndarray, delta: float = 1.0) -> np.ndarray:
        if self.gating_model is None:
            raise ValueError("Gating model not initialised")
        probs = self.gating_model.predict_proba(X)
        out = np.zeros_like(X)
        for k, (L, bias) in enumerate(zip(self.L_list, self.bias_list)):
            branch_flow = pushforward(X, L, tau=delta) + bias
            out += probs[:, [k]] * branch_flow
        return out

    def forward_steps(self, X: np.ndarray, steps: int = 2, delta: float = 1.0) -> np.ndarray:
        result = X
        for _ in range(steps):
            result = self.forward(result, delta=delta)
        return result

    @property
    def stability_penalty(self) -> float:
        margin = 0.0 if self.stable_margin is None else float(self.stable_margin)
        if self.log_diagnostics and self.stability_diagnostics:
            mu2_values = [entry.get("mu2_raw", 0.0) for entry in self.stability_diagnostics]
            return stability_penalty(mu2_values, margin=margin, lam=self.stable_lambda)
        return stability_penalty(self.L_list, margin=margin, lam=self.stable_lambda)


def semigroup_consistency(
    model: GatedSemigroup,
    X0: np.ndarray,
    Xt1: np.ndarray,
    Xt2: np.ndarray,
    Xt12: np.ndarray,
    reg: float,
    *,
    clip_eps: float = 1e-8,
    eval_disable_projection: bool = False,
) -> SemigroupReport:
    pred_mid = model.forward(X0, delta=1.0)
    pred_two = model.forward(pred_mid, delta=1.0)

    mid_error = sinkhorn_divergence_bal(
        pred_mid,
        Xt1,
        reg=reg,
        backend=model.ot_backend,
        sinkhorn_backend=model.sinkhorn_backend,
        scaling=model.sinkhorn_scaling,
        num_iter_max=model.num_iter_max,
        stop_thr=model.stop_thr,
    )
    two_error = sinkhorn_divergence_bal(
        pred_two,
        Xt12,
        reg=reg,
        backend=model.ot_backend,
        sinkhorn_backend=model.sinkhorn_backend,
        scaling=model.sinkhorn_scaling,
        num_iter_max=model.num_iter_max,
        stop_thr=model.stop_thr,
    )

    direct_model = GatedSemigroup(
        K=model.K,
        rank=model.rank,
        reg=model.reg,
        reg_m=model.reg_m,
        stable_margin=None if eval_disable_projection else model.stable_margin,
        stable_alpha=model.stable_alpha,
        stable_lambda=model.stable_lambda,
        stable_soft_weight=model.stable_soft_weight,
        stable_use_soft_penalty=model.stable_use_soft_penalty,
        stable_lambda_soft=model.stable_lambda_soft,
        use_swa=model.use_swa,
        swa_start_ratio=model.swa_start_ratio,
        swa_lr=model.swa_lr,
        swa_schedule=model.swa_schedule,
        ot_backend=model.ot_backend,
        sinkhorn_backend=model.sinkhorn_backend,
        sinkhorn_scaling=model.sinkhorn_scaling,
        min_branch_samples=model.min_branch_samples,
        tau=model.tau,
        num_iter_max=model.num_iter_max,
        stop_thr=model.stop_thr,
        train_steps=model.train_steps,
        train_lr=model.train_lr,
        reg_nuc=model.reg_nuc,
        use_torch=model.use_torch,
        device=model.device,
        log_diagnostics=model.log_diagnostics,
        dynamics_cfg=model.dynamics_cfg,
        regular_cfg=model.regular_cfg,
        align_cfg=model.align_cfg,
    )
    direct_model.gating_model = model.gating_model
    direct_model.fit_pair(X0, Xt12)
    direct_pred = direct_model.forward(X0, delta=1.0)
    direct_error = sinkhorn_divergence_bal(
        direct_pred,
        Xt12,
        reg=reg,
        backend=model.ot_backend,
        sinkhorn_backend=model.sinkhorn_backend,
        scaling=model.sinkhorn_scaling,
        num_iter_max=model.num_iter_max,
        stop_thr=model.stop_thr,
    )

    consistency_error = sinkhorn_divergence_bal(
        pred_two,
        direct_pred,
        reg=reg,
        backend=model.ot_backend,
        sinkhorn_backend=model.sinkhorn_backend,
        scaling=model.sinkhorn_scaling,
        num_iter_max=model.num_iter_max,
        stop_thr=model.stop_thr,
    )

    mid_val = max(float(mid_error), clip_eps)
    two_val = max(float(two_error), clip_eps)
    direct_val = max(float(direct_error), clip_eps)
    cons_val = max(float(consistency_error), clip_eps)
    improvement = _consistency_improvement(direct_val, two_val, clip_eps)

    total_penalty = model.stability_penalty + direct_model.stability_penalty

    return SemigroupReport(
        mid_error=mid_val,
        two_step_error=two_val,
        direct_error=direct_val,
        consistency_error=cons_val,
        improvement=float(improvement),
        stability_penalty=float(total_penalty),
    )


__all__ = [
    "GatedSemigroup",
    "SemigroupReport",
    "semigroup_consistency",
]
