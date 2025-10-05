# -*- coding: utf-8 -*-
"""Direct generator training with gradient descent and PyTorch optimization."""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TORCH = False
    torch = None  # type: ignore

# Detect GeomLoss availability at module import for auto Torch selection on CPU
try:  # pragma: no cover - optional dependency
    from geomloss import SamplesLoss as _GL_SamplesLoss  # noqa: F401
    _GEOMLOSS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _GEOMLOSS_AVAILABLE = False

from ..ot.uot_losses import uot_sinkhorn_cost
from ..ot import uot_losses as _uot_losses
from ..stability import project_to_stable, spectral_abscissa, mu2_log_norm, soft_stability_penalty
from .semigroup import project_stable, pushforward

Array = np.ndarray


def fit_branch_generator(
    Xs: Array,
    Xt: Array,
    *,
    tau: float = 0.5,
    rank: int | None = None,
    steps: int = 200,
    lr: float = 1e-2,
    alpha: float = 1e-3,
    reg_nuc: float = 0.0,
    seed: int = 0,
    use_torch: bool | None = None,
    use_torch_compile: bool | None = True,
    device: str = "cpu",
    reg: float = 0.08,
    reg_m: float = 1.0,
    num_iter_max: int = 1000,
    stop_thr: float = 1e-6,
    stable_margin: float | None = -1e-3,
    soft_weight: float | None = None,
    use_soft_penalty: bool | None = None,
    lambda_soft: float = 0.5,
    use_swa: bool = False,
    swa_start_ratio: float = 0.8,
    swa_lr: float = 1e-3,
    swa_schedule: str = "constant",
    log_diagnostics: bool | None = None,
    backend: str = "geomloss",
    sinkhorn_backend: str | None = None,
    sinkhorn_scaling: float = 0.9,
    sinkhorn_minibatch: bool = False,
    sinkhorn_batch_size: int | None = None,
    model: str = "linear",
    K: int = 2,
    residual_cfg: dict[str, Any] | None = None,
    meta_cfg: dict[str, Any] | None = None,
    rk: str = "rk2",
    jac_penalty_weight: float = 0.0,
) -> Array | tuple[Array, dict[str, float]]:
    """Fit branch-specific generator using POT/GeomLoss objective."""

    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError(f"Dimension mismatch: Xs{Xs.shape} vs Xt{Xt.shape}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    sink_backend = sinkhorn_backend or "online"
    soft_val = float(soft_weight if soft_weight is not None else 0.5)
    soft_val = max(0.0, min(1.0, soft_val))
    soft_penalty_flag = True if use_soft_penalty is None else bool(use_soft_penalty)
    lambda_soft_val = float(lambda_soft)
    use_swa_flag = bool(use_swa)
    swa_start = float(swa_start_ratio)
    swa_lr_val = float(swa_lr)
    swa_schedule_val = swa_schedule or 'constant'
    capture_diag = bool(log_diagnostics)

    # Auto-enable Torch on CPU if GeomLoss+Torch are available
    if use_torch is None:
        use_torch = bool(HAS_TORCH and _GEOMLOSS_AVAILABLE)
    if use_torch and not HAS_TORCH:
        warnings.warn("PyTorch not available, falling back to NumPy implementation")
        use_torch = False
    if use_torch:
        if backend != "geomloss":
            warnings.warn("PyTorch optimisation requires backend='geomloss'; falling back to NumPy implementation")
            use_torch = False
        else:
            try:
                result = _fit_branch_generator_torch(
                    Xs,
                    Xt,
                    tau=tau,
                    rank=rank,
                    steps=steps,
                    lr=lr,
                    alpha=alpha,
                    reg_nuc=reg_nuc,
                    seed=seed,
                    reg=reg,
                    reg_m=reg_m,
                    stable_margin=stable_margin,
                    soft_weight=soft_val,
                    use_soft_penalty=soft_penalty_flag,
                    lambda_soft=lambda_soft_val,
                    use_swa=use_swa_flag,
                    swa_start_ratio=swa_start,
                    swa_lr=swa_lr_val,
                    swa_schedule=swa_schedule_val,
                    device=device,
                    capture_diag=bool(log_diagnostics),
                    sinkhorn_backend=sink_backend,
                    sinkhorn_scaling=sinkhorn_scaling,
                    use_torch_compile=bool(use_torch_compile),
                    sinkhorn_minibatch=bool(sinkhorn_minibatch),
                    sinkhorn_batch_size=sinkhorn_batch_size,
                    model=model,
                    K=K,
                    residual_cfg=residual_cfg,
                    meta_cfg=meta_cfg,
                    rk=rk,
                    jac_penalty_weight=float(jac_penalty_weight),
                )
                if result is not None:
                    L_torch, diag_torch = result
                    if log_diagnostics:
                        return L_torch, diag_torch
                    return L_torch
            except Exception as exc:  # pragma: no cover - safety fallback
                warnings.warn(f"PyTorch path failed ({exc!r}); falling back to NumPy implementation")
                use_torch = False


    L_fit, diagnostics = _fit_branch_generator_numpy(
        Xs,
        Xt,
        tau=tau,
        rank=rank,
        steps=steps,
        lr=lr,
        alpha=alpha,
        reg_nuc=reg_nuc,
        seed=seed,
        reg=reg,
        reg_m=reg_m,
        num_iter_max=num_iter_max,
        stop_thr=stop_thr,
        stable_margin=stable_margin,
        soft_weight=soft_val,
        use_soft_penalty=soft_penalty_flag,
        lambda_soft=lambda_soft_val,
        use_swa=use_swa_flag,
        swa_start_ratio=swa_start,
        swa_lr=swa_lr_val,
        swa_schedule=swa_schedule_val,
        capture_diag=capture_diag,
        backend=backend,
        sinkhorn_backend=sink_backend,
        sinkhorn_scaling=sinkhorn_scaling,
    )
    if capture_diag:
        return L_fit, diagnostics
    return L_fit


def _fit_branch_generator_numpy(
    Xs: Array,
    Xt: Array,
    tau: float,
    rank: Optional[int],
    steps: int,
    lr: float,
    alpha: float,
    reg_nuc: float,
    seed: int,
    reg: float,
    reg_m: float,
    num_iter_max: int,
    stop_thr: float,
    stable_margin: float | None,
    soft_weight: float,
    use_soft_penalty: bool,
    lambda_soft: float,
    use_swa: bool,
    swa_start_ratio: float,
    swa_lr: float,
    swa_schedule: str,
    capture_diag: bool,
    backend: str,
    sinkhorn_backend: str,
    sinkhorn_scaling: float,
) -> tuple[Array, dict[str, float]]:
    rng = np.random.default_rng(seed)
    max_samples = int(os.environ.get('CT_OTS_MAX_TRAIN_SAMPLES', 80))
    if Xs.shape[0] > max_samples:
        idx_src = rng.choice(Xs.shape[0], max_samples, replace=False)
        Xs = Xs[idx_src]
    if Xt.shape[0] > max_samples:
        idx_tgt = rng.choice(Xt.shape[0], max_samples, replace=False)
        Xt = Xt[idx_tgt]

    d = Xs.shape[1]
    r = rank or min(32, d)

    U = rng.standard_normal((d, r)) * 0.01
    V = rng.standard_normal((d, r)) * 0.01

    lr_adapt = lr
    best_loss = np.inf
    best_L: Array | None = None
    best_diag: dict[str, float] | None = None
    patience = 20
    no_improve = 0

    projection_enabled = stable_margin is not None
    margin_value = 0.0 if stable_margin is None else float(stable_margin)
    alpha_penalty = abs(margin_value) if margin_value != 0.0 else abs(alpha)

    use_swa_phase = use_swa and steps > 1
    swa_start_step = max(int(round(swa_start_ratio * steps)), 0)
    swa_sum: Array | None = None
    swa_count = 0

    swa_schedule_lower = (swa_schedule or 'constant').lower()

    warm_potentials: tuple[np.ndarray | None, np.ndarray | None] | None = None

    def compute_objective(U_mat: Array, V_mat: Array, *, allow_warmstart: bool = True):
        nonlocal warm_potentials
        L_raw = U_mat @ V_mat.T
        alpha_raw = spectral_abscissa(L_raw)
        mu2_raw = mu2_log_norm(L_raw)

        L_stable = project_stable(L_raw, alpha=alpha)
        L_proj, diag = project_to_stable(
            L_stable,
            margin=margin_value,
            enable=projection_enabled,
            soft_weight=soft_weight,
        )
        diag['alpha_raw'] = alpha_raw
        diag['mu2_raw'] = mu2_raw
        diag['violation'] = max(0.0, mu2_raw - margin_value)
        diag['margin'] = margin_value

        Yhat = pushforward(Xs, L_proj, tau=tau)
        if backend == 'geomloss':
            loss_uot, _, _ = uot_sinkhorn_cost(
                Yhat,
                Xt,
                reg=reg,
                reg_m=reg_m,
                metric='sqeuclidean',
                num_iter_max=num_iter_max,
                stop_thr=stop_thr,
                backend=backend,
                sinkhorn_backend=sinkhorn_backend,
                scaling=sinkhorn_scaling,
            )
        else:
            if allow_warmstart and warm_potentials is not None:
                ws = warm_potentials
            else:
                ws = None
            if allow_warmstart:
                loss_uot, _, _, next_ws = uot_sinkhorn_cost(
                    Yhat,
                    Xt,
                    reg=reg,
                    reg_m=reg_m,
                    metric='sqeuclidean',
                    num_iter_max=num_iter_max,
                    stop_thr=stop_thr,
                    backend=backend,
                    sinkhorn_backend=sinkhorn_backend,
                    scaling=sinkhorn_scaling,
                    warm_start=ws,
                    return_warm_start=True,
                )
                warm_potentials = next_ws
            else:
                loss_uot, _, _ = uot_sinkhorn_cost(
                    Yhat,
                    Xt,
                    reg=reg,
                    reg_m=reg_m,
                    metric='sqeuclidean',
                    num_iter_max=num_iter_max,
                    stop_thr=stop_thr,
                    backend=backend,
                    sinkhorn_backend=sinkhorn_backend,
                    scaling=sinkhorn_scaling,
                )
        loss_nuc = reg_nuc * np.linalg.norm(L_proj, ord='nuc') if reg_nuc else 0.0
        soft_pen = (
            soft_stability_penalty(L_raw, alpha=alpha_penalty, lambda_stab=lambda_soft)
            if use_soft_penalty
            else 0.0
        )
        total = float(loss_uot + loss_nuc + soft_pen)
        diag['loss_uot'] = float(loss_uot)
        diag['loss_nuc'] = float(loss_nuc)
        diag['soft_penalty'] = float(soft_pen)
        diag['loss_total'] = total
        return total, L_proj, diag, Yhat

    def get_swa_lr(iter_idx: int) -> float:
        if swa_schedule_lower == 'cosine':
            denom = max(1, steps - swa_start_step)
            progress = (iter_idx - swa_start_step) / denom
            progress = min(max(progress, 0.0), 1.0)
            return swa_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        return swa_lr

    final_diag: dict[str, float] | None = None

    for it in range(steps):
        total_loss, L_proj, diag, Yhat = compute_objective(U, V)
        final_diag = diag if capture_diag else None

        if total_loss < best_loss:
            best_loss = total_loss
            best_L = L_proj.copy()
            best_diag = diag.copy() if capture_diag else None
            no_improve = 0
        else:
            no_improve += 1

        in_swa = use_swa_phase and it >= swa_start_step
        if in_swa:
            if swa_sum is None:
                swa_sum = L_proj.copy()
                swa_count = 1
            else:
                swa_sum += L_proj
                swa_count += 1

        if not in_swa and no_improve > patience:
            lr_adapt *= 0.8
            no_improve = 0
            if lr_adapt < 1e-6:
                break

        eps = max(1e-4, lr_adapt * 0.1)
        G_U = np.zeros_like(U)
        G_V = np.zeros_like(V)

        n_samples = min(r * d // 4, 50)
        for _ in range(n_samples):
            i = rng.integers(0, d)
            j = rng.integers(0, r)

            U_plus = U.copy()
            U_plus[i, j] += eps
            loss_plus, _, _, _ = compute_objective(U_plus, V, allow_warmstart=False)

            U_minus = U.copy()
            U_minus[i, j] -= eps
            loss_minus, _, _, _ = compute_objective(U_minus, V, allow_warmstart=False)
            G_U[i, j] = (loss_plus - loss_minus) / (2 * eps)

            V_plus = V.copy()
            V_plus[i, j] += eps
            loss_plus_v, _, _, _ = compute_objective(U, V_plus, allow_warmstart=False)

            V_minus = V.copy()
            V_minus[i, j] -= eps
            loss_minus_v, _, _, _ = compute_objective(U, V_minus, allow_warmstart=False)
            G_V[i, j] = (loss_plus_v - loss_minus_v) / (2 * eps)

        lr_step = get_swa_lr(it) if in_swa else lr_adapt
        U -= lr_step * G_U
        V -= lr_step * G_V

        if it % 50 == 0:
            print(
                f"NumPy Step {it}: total={total_loss:.6f}, "
                f"best={best_loss:.6f}, lr={lr_step:.6f}, "
                f"soft_pen={diag.get('soft_penalty', 0.0):.6f}"
            )

    if use_swa_phase and swa_count > 0 and swa_sum is not None:
        swa_avg = swa_sum / swa_count
        swa_stable = project_stable(swa_avg, alpha=alpha)
        swa_proj, swa_diag = project_to_stable(
            swa_stable,
            margin=margin_value,
            enable=projection_enabled,
            soft_weight=soft_weight,
        )
        Yhat_swa = pushforward(Xs, swa_proj, tau=tau)
        loss_uot_swa, _, _ = uot_sinkhorn_cost(
            Yhat_swa,
            Xt,
            reg=reg,
            reg_m=reg_m,
            metric="sqeuclidean",
            num_iter_max=num_iter_max,
            stop_thr=stop_thr,
            backend=backend,
            sinkhorn_backend=sinkhorn_backend,
            scaling=sinkhorn_scaling,
        )
        loss_nuc_swa = reg_nuc * np.linalg.norm(swa_proj, ord="nuc") if reg_nuc else 0.0
        soft_pen_swa = soft_stability_penalty(swa_avg, alpha=alpha_penalty, lambda_stab=lambda_soft) if use_soft_penalty else 0.0
        total_swa = float(loss_uot_swa + loss_nuc_swa + soft_pen_swa)
        swa_diag['alpha_raw'] = spectral_abscissa(swa_avg)
        swa_diag['mu2_raw'] = mu2_log_norm(swa_avg)
        swa_diag['violation'] = max(0.0, swa_diag['mu2_raw'] - margin_value)
        swa_diag['margin'] = margin_value
        swa_diag['loss_uot'] = float(loss_uot_swa)
        swa_diag['loss_nuc'] = float(loss_nuc_swa)
        swa_diag['soft_penalty'] = float(soft_pen_swa)
        swa_diag['loss_total'] = total_swa
        swa_diag['swa_updates'] = float(swa_count)
        if total_swa <= best_loss:
            best_loss = total_swa
            best_L = swa_proj
            if capture_diag:
                best_diag = swa_diag

    if best_L is None:
        _, L_proj, diag, _ = compute_objective(U, V)
        best_L = L_proj
        if capture_diag:
            best_diag = diag
    else:
        best_diag = best_diag or final_diag or {}

    if not capture_diag:
        best_diag = {}

    return best_L, best_diag or {}


def fit_low_rank_generator(
    Xs: Array,
    Xt: Array,
    rank: int,
    *,
    tau: float = 0.5,
    steps: int = 200,
    lr: float = 1e-2,
    alpha: float = 1e-3,
    seed: int = 0,
    reg: float = 0.08,
    reg_m: float = 1.0,
    num_iter_max: int = 1000,
    stop_thr: float = 1e-6,
    stable_margin: float | None = -1e-3,
    soft_weight: float | None = None,
    backend: str = "geomloss",
    sinkhorn_backend: str | None = None,
    sinkhorn_scaling: float = 0.9,
) -> Tuple[Array, Array, Array]:
    rng = np.random.default_rng(seed)
    max_samples = int(os.environ.get('CT_OTS_MAX_TRAIN_SAMPLES', 80))
    if Xs.shape[0] > max_samples:
        idx_src = rng.choice(Xs.shape[0], max_samples, replace=False)
        Xs = Xs[idx_src]
    if Xt.shape[0] > max_samples:
        idx_tgt = rng.choice(Xt.shape[0], max_samples, replace=False)
        Xt = Xt[idx_tgt]

    d = Xs.shape[1]

    U = rng.standard_normal((d, rank)) * 0.01
    V = rng.standard_normal((d, rank)) * 0.01

    best_loss = np.inf
    best_U, best_V = U.copy(), V.copy()
    margin_value = 0.0 if stable_margin is None else float(stable_margin)
    sink_backend = sinkhorn_backend or "online"

    soft_val = float(soft_weight if soft_weight is not None else 0.5)
    warm_potentials: tuple[np.ndarray | None, np.ndarray | None] | None = None

    for it in range(steps):
        L_stable = project_stable(U @ V.T, alpha=alpha)
        L, _ = project_to_stable(
            L_stable,
            margin=margin_value,
            enable=stable_margin is not None,
            soft_weight=soft_val,
        )
        Yhat = pushforward(Xs, L, tau=tau)
        if backend == 'geomloss':
            loss, _, _ = uot_sinkhorn_cost(
                Yhat,
                Xt,
                reg=reg,
                reg_m=reg_m,
                metric="sqeuclidean",
                num_iter_max=num_iter_max,
                stop_thr=stop_thr,
                backend=backend,
                sinkhorn_backend=sink_backend,
                scaling=sinkhorn_scaling,
            )
        else:
            loss, _, _, warm_potentials = uot_sinkhorn_cost(
                Yhat,
                Xt,
                reg=reg,
                reg_m=reg_m,
                metric="sqeuclidean",
                num_iter_max=num_iter_max,
                stop_thr=stop_thr,
                backend=backend,
                sinkhorn_backend=sink_backend,
                scaling=sinkhorn_scaling,
                warm_start=warm_potentials,
                return_warm_start=True,
            )

        if loss < best_loss:
            best_loss = loss
            best_U, best_V = U.copy(), V.copy()

        eps = 1e-4
        G_U = np.zeros_like(U)
        G_V = np.zeros_like(V)

        L_current = L
        Y_current = Yhat

        for i in range(d):
            for j in range(rank):
                U_pert = U.copy()
                U_pert[i, j] += eps
                L_pert = project_stable(U_pert @ V.T, alpha=alpha)
                L_pert, _ = project_to_stable(
                    L_pert,
                    margin=margin_value,
                    enable=stable_margin is not None,
                    soft_weight=soft_val,
                )
                Y_pert = pushforward(Xs, L_pert, tau=tau)
                loss_pert, _, _ = uot_sinkhorn_cost(
                    Y_pert,
                    Xt,
                    reg=reg,
                    reg_m=reg_m,
                    metric="sqeuclidean",
                    num_iter_max=num_iter_max,
                    stop_thr=stop_thr,
                    backend=backend,
                    sinkhorn_backend=sink_backend,
                    scaling=sinkhorn_scaling,
                )
                loss_current, _, _ = uot_sinkhorn_cost(
                    Y_current,
                    Xt,
                    reg=reg,
                    reg_m=reg_m,
                    metric="sqeuclidean",
                    num_iter_max=num_iter_max,
                    stop_thr=stop_thr,
                    backend=backend,
                    sinkhorn_backend=sink_backend,
                    scaling=sinkhorn_scaling,
                )
                G_U[i, j] = (loss_pert - loss_current) / eps

        for i in range(d):
            for j in range(rank):
                V_pert = V.copy()
                V_pert[i, j] += eps
                L_pert = project_stable(U @ V_pert.T, alpha=alpha)
                L_pert, _ = project_to_stable(
                    L_pert,
                    margin=margin_value,
                    enable=stable_margin is not None,
                    soft_weight=soft_val,
                )
                Y_pert = pushforward(Xs, L_pert, tau=tau)
                loss_pert, _, _ = uot_sinkhorn_cost(
                    Y_pert,
                    Xt,
                    reg=reg,
                    reg_m=reg_m,
                    metric="sqeuclidean",
                    num_iter_max=num_iter_max,
                    stop_thr=stop_thr,
                    backend=backend,
                    sinkhorn_backend=sink_backend,
                    scaling=sinkhorn_scaling,
                )
                loss_current, _, _ = uot_sinkhorn_cost(
                    Y_current,
                    Xt,
                    reg=reg,
                    reg_m=reg_m,
                    metric="sqeuclidean",
                    num_iter_max=num_iter_max,
                    stop_thr=stop_thr,
                    backend=backend,
                    sinkhorn_backend=sink_backend,
                    scaling=sinkhorn_scaling,
                )
                G_V[i, j] = (loss_pert - loss_current) / eps

        U -= lr * G_U
        V -= lr * G_V

    L_final_stable = project_stable(best_U @ best_V.T, alpha=alpha)
    L_final, _ = project_to_stable(
        L_final_stable,
        margin=margin_value,
        enable=stable_margin is not None,
    )
    return best_U, best_V, L_final


def _torch_safe_eigh(mat: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    try:
        return torch.linalg.eigh(mat)
    except RuntimeError:
        eye = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        try:
            return torch.linalg.eigh(mat + eye * 1e-6)
        except RuntimeError:
            mat64 = (mat + eye * 1e-6).to(torch.float64)
            w, Q = torch.linalg.eigh(mat64)
            return w.to(mat.dtype), Q.to(mat.dtype)


def _torch_safe_eigvalsh(mat: "torch.Tensor") -> "torch.Tensor":
    try:
        return torch.linalg.eigvalsh(mat)
    except RuntimeError:
        mat64 = mat.to(torch.float64)
        vals = torch.linalg.eigvalsh(mat64)
        return vals.to(mat.dtype)


def _torch_max_real_eig(mat: "torch.Tensor") -> float:
    try:
        vals = torch.linalg.eigvals(mat)
    except RuntimeError:
        vals = torch.linalg.eigvals(mat.to(torch.float64))
    return float(torch.real(vals).max().item())


def _torch_project_stable(L: "torch.Tensor", alpha: float) -> "torch.Tensor":
    S = 0.5 * (L + L.T)
    S = 0.5 * (S + S.T)
    w, Q = _torch_safe_eigh(S)
    clip_val = -abs(float(alpha)) if alpha is not None else 0.0
    w_clipped = torch.clamp(w, max=clip_val)
    S_proj = (Q * w_clipped.unsqueeze(0)) @ Q.T
    A = 0.5 * (L - L.T)
    return S_proj + A


def _torch_project_to_stable(
    L: "torch.Tensor",
    margin: float | None,
    soft_weight: float,
) -> tuple["torch.Tensor", dict[str, float]]:
    margin_value = 0.0 if margin is None else float(margin)
    blend = float(np.clip(soft_weight, 0.0, 1.0))
    L_proj = L

    if margin is not None:
        S = 0.5 * (L + L.T)
        S = 0.5 * (S + S.T)
        w, Q = _torch_safe_eigh(S)
        w_clipped = torch.clamp(w, max=margin_value)
        S_proj = (Q * w_clipped.unsqueeze(0)) @ Q.T
        A = 0.5 * (L - L.T)
        hard = S_proj + A
        if blend < 1.0:
            L_proj = L + blend * (hard - L)
        else:
            L_proj = hard

    alpha_raw = _torch_max_real_eig(L)
    mu2_raw = float(_torch_safe_eigvalsh(0.5 * (L + L.T)).max().item())
    mu2_proj = float(_torch_safe_eigvalsh(0.5 * (L_proj + L_proj.T)).max().item())
    diag = {
        'alpha_raw': alpha_raw,
        'mu2_raw': mu2_raw,
        'alpha_proj': mu2_proj if margin is not None else alpha_raw,
        'mu2_proj': mu2_proj,
        'violation': max(0.0, mu2_raw - margin_value),
        'margin': margin_value,
        'soft_weight': blend,
    }
    return L_proj, diag


def _torch_soft_stability_penalty(L: "torch.Tensor", alpha: float, lambda_stab: float) -> "torch.Tensor":
    S = 0.5 * (L + L.T)
    mu2 = torch.linalg.eigvalsh(S).max()
    violation = torch.clamp(mu2 + abs(float(alpha)), min=0.0)
    return lambda_stab * violation.square()




def _fit_branch_generator_torch(
    Xs: Array,
    Xt: Array,
    *,
    tau: float,
    rank: Optional[int],
    steps: int,
    lr: float,
    alpha: float,
    reg_nuc: float,
    seed: int,
    reg: float,
    reg_m: float,
    stable_margin: float | None,
    soft_weight: float,
    use_soft_penalty: bool,
    lambda_soft: float,
    use_swa: bool,
    swa_start_ratio: float,
    swa_lr: float,
    swa_schedule: str,
    device: str,
    capture_diag: bool,
    sinkhorn_backend: str,
    sinkhorn_scaling: float,
    use_torch_compile: bool,
    sinkhorn_minibatch: bool,
    sinkhorn_batch_size: int | None,
    model: str,
    K: int,
    residual_cfg: dict[str, Any] | None,
    meta_cfg: dict[str, Any] | None,
    rk: str,
    jac_penalty_weight: float,
) -> tuple[Array, dict[str, float]] | None:
    if not HAS_TORCH:
        return None
    try:
        from geomloss import SamplesLoss  # noqa: F401
    except ImportError:  # pragma: no cover - optional dependency
        warnings.warn('geomloss is required for PyTorch optimisation; falling back to NumPy implementation')
        return None

    from .modules.stable_linear import StableLinear
    from .modules.residuals import ResidualNet
    from .modules.meta_head import MetaHead
    from ..transport.pushforward import pushforward_torch
    from .ode.jacobian_penalty import approx_jacobian_spectral_norm

    if Xs.shape[0] == 0 or Xt.shape[0] == 0:
        return None

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    max_samples = int(os.environ.get('CT_OTS_MAX_TRAIN_SAMPLES', 80))
    src_sample_indices: np.ndarray | None = None
    if Xs.shape[0] > max_samples:
        src_sample_indices = rng.choice(Xs.shape[0], max_samples, replace=False)
        Xs = Xs[src_sample_indices]
    if Xt.shape[0] > max_samples:
        Xt = Xt[rng.choice(Xt.shape[0], max_samples, replace=False)]

    try:
        requested_device = torch.device(device)
    except RuntimeError:
        requested_device = torch.device('cpu')
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        warnings.warn('CUDA requested but not available; using CPU')
        requested_device = torch.device('cpu')

    dtype = torch.float32
    Xs_t = torch.as_tensor(Xs, dtype=dtype, device=requested_device)
    Xt_t = torch.as_tensor(Xt, dtype=dtype, device=requested_device)

    d = Xs_t.shape[1]
    base_rank = rank or min(32, d)
    if meta_cfg and 'rank' in meta_cfg:
        base_rank = int(meta_cfg.get('rank', base_rank))
    r = max(1, int(base_rank))

    gamma_init = abs(float(stable_margin)) if stable_margin is not None else abs(float(alpha))
    if gamma_init <= 0:
        gamma_init = 1e-3

    flow_mode = (model or 'linear').lower()
    if flow_mode not in {'linear', 'exp_rnn', 'ode'}:
        raise ValueError(f"Unsupported model '{model}'")

    L_module = StableLinear(d, r=r, gamma_init=gamma_init, device=requested_device, dtype=dtype)
    parameters: list[torch.nn.Parameter] = list(L_module.parameters())

    residual = None
    if flow_mode in {'exp_rnn', 'ode'}:
        rcfg: dict[str, Any] = {'hidden': 128, 'depth': 2, 'lipschitz_scale': 1.0, 'activation': 'relu'}
        if residual_cfg:
            rcfg.update(residual_cfg)
        residual = ResidualNet(d, **rcfg).to(device=requested_device, dtype=dtype)
        parameters += list(residual.parameters())

    meta_head = None
    meta_context = None
    meta_aggregate = 'mean'
    if meta_cfg:
        meta_aggregate = str(meta_cfg.get('aggregate', 'mean')).lower()
        context = meta_cfg.get('context')
        if context is None:
            raise ValueError("meta_cfg requires a 'context' entry")
        if src_sample_indices is not None:
            if isinstance(context, torch.Tensor):
                index_tensor = torch.as_tensor(
                    src_sample_indices,
                    device=context.device if context.device.type != 'meta' else 'cpu',
                    dtype=torch.long,
                )
                context = context.index_select(0, index_tensor)
            else:
                context = np.asarray(context)[src_sample_indices]
        if isinstance(context, torch.Tensor):
            meta_context = context.to(device=requested_device, dtype=dtype)
        else:
            meta_context = torch.as_tensor(context, device=requested_device, dtype=dtype)
        if meta_context.shape[0] != Xs_t.shape[0]:
            raise ValueError('meta context must align with source samples')
        context_dim = int(meta_context.shape[1])
        meta_head = MetaHead(context_dim, d, r).to(device=requested_device, dtype=dtype)
        parameters += list(meta_head.parameters())

    optimizer = torch.optim.AdamW(parameters, lr=lr, amsgrad=True)

    attempts = _uot_losses._geomloss_candidates(sinkhorn_backend, sinkhorn_scaling)
    sinkhorn = None
    last_exc: Exception | None = None
    for candidate_backend, candidate_scaling in attempts:
        try:
            sinkhorn = _uot_losses._make_samples_loss(
                blur=reg,
                debias=True,
                backend=candidate_backend,
                scaling=candidate_scaling,
                reach=reg_m,
            )
            with torch.no_grad():
                probe_x = Xs_t[: min(Xs_t.shape[0], 32)]
                probe_y = Xt_t[: min(Xt_t.shape[0], 32)]
                sinkhorn(probe_x, probe_y)
            break
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_exc = exc
            sinkhorn = None
            continue
    if sinkhorn is None:
        warnings.warn(
            f"GeomLoss SamplesLoss failed to initialise for PyTorch backend; falling back to NumPy implementation ({last_exc!r})"
        )
        return None

    def _select(tensor: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
        return tensor if idx is None else tensor.index_select(0, idx)

    def _meta_overrides(idx: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if meta_head is None or meta_context is None:
            return None
        context_batch = _select(meta_context, idx)
        U_b, V_b, W_b, logg_b = meta_head(context_batch)
        if meta_aggregate == 'mean':
            U = U_b.mean(dim=0)
            V = V_b.mean(dim=0)
            W = W_b.mean(dim=0)
            logg = logg_b.mean(dim=0)
        elif meta_aggregate == 'first':
            U = U_b[0]
            V = V_b[0]
            W = W_b[0]
            logg = logg_b[0]
        else:
            raise ValueError(f"Unsupported meta aggregate '{meta_aggregate}'")
        return U, V, W, logg

    def _make_L(idx: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        overrides = _meta_overrides(idx)
        L_raw = L_module(overrides)
        L_stable = _torch_project_stable(L_raw, alpha)
        L_proj, diag = _torch_project_to_stable(L_stable, stable_margin, soft_weight)
        return L_raw, L_proj, diag

    alpha_penalty = abs(float(stable_margin)) if stable_margin is not None else abs(float(alpha))

    def _jacobian_penalty(batch: torch.Tensor) -> torch.Tensor:
        if jac_penalty_weight <= 0.0 or residual is None:
            return torch.zeros((), device=batch.device, dtype=batch.dtype)
        stride = max(1, batch.shape[0] // 8)
        sample = batch[::stride] if stride > 0 else batch
        if sample.shape[0] == 0:
            sample = batch[:1]
        return jac_penalty_weight * approx_jacobian_spectral_norm(residual, sample, iters=1)

    exp_cache: Dict[object, torch.Tensor] = {}

    def _loss_fn(idx_s: torch.Tensor | None = None, idx_t: torch.Tensor | None = None) -> torch.Tensor:
        Xs_used = _select(Xs_t, idx_s)
        Xt_used = _select(Xt_t, idx_t)
        L_raw, L_proj, _ = _make_L(idx_s)
        Yhat = pushforward_torch(
            Xs_used,
            L_module,
            tau,
            mode=flow_mode,
            residual=residual,
            K=K,
            rk=rk,
            cache=exp_cache,
            L_override=L_proj,
        )
        loss_transport = sinkhorn(Yhat, Xt_used)
        loss_total = loss_transport
        if reg_nuc:
            loss_total = loss_total + reg_nuc * torch.linalg.matrix_norm(L_proj, ord='nuc')
        if use_soft_penalty:
            loss_total = loss_total + _torch_soft_stability_penalty(L_raw, alpha_penalty, lambda_soft)
        if jac_penalty_weight > 0.0 and residual is not None:
            loss_total = loss_total + _jacobian_penalty(Xs_used)
        return loss_total

    compiled_loss = _loss_fn
    if use_torch_compile and hasattr(torch, "compile"):
        try:  # pragma: no cover - optional optimisation path
            compiled_loss = torch.compile(_loss_fn, mode="reduce-overhead")  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - runtime dependent
            warnings.warn(f"torch.compile unavailable or failed ({exc!r}); continuing without compilation")

    best_loss = torch.tensor(float('inf'), device=requested_device)
    best_L: torch.Tensor | None = None
    best_diag: dict[str, float] | None = None

    use_swa_phase = bool(use_swa) and steps > 1
    swa_start_step = max(int(round(swa_start_ratio * steps)), 0)
    swa_sum: torch.Tensor | None = None
    swa_count = 0
    swa_schedule_lower = (swa_schedule or 'constant').lower()

    def get_swa_lr(iter_idx: int) -> float:
        if swa_schedule_lower == 'cosine':
            denom = max(1, steps - swa_start_step)
            progress = min(max((iter_idx - swa_start_step) / denom, 0.0), 1.0)
            return float(swa_lr * 0.5 * (1.0 + np.cos(np.pi * progress)))
        return float(swa_lr)

    margin_value = 0.0 if stable_margin is None else float(stable_margin)

    for step_idx in range(steps):
        optimizer.zero_grad()
        idx_s = idx_t = None
        if sinkhorn_minibatch:
            bs = sinkhorn_batch_size or int(min(1024, int(Xs_t.shape[0]), int(Xt_t.shape[0])))
            if bs > 0:
                idx_s = torch.randint(0, int(Xs_t.shape[0]), (bs,), device=requested_device)
                idx_t = torch.randint(0, int(Xt_t.shape[0]), (bs,), device=requested_device)
        try:
            loss_total = compiled_loss(idx_s, idx_t)
        except Exception as exc:  # pragma: no cover - runtime dependent
            warnings.warn(f"torch.compile execution failed ({exc!r}); disabling compilation for remaining steps")
            compiled_loss = _loss_fn
            loss_total = compiled_loss(idx_s, idx_t)

        with torch.no_grad():
            L_tmp_raw, L_tmp_proj, diag = _make_L(None)
            loss_uot_dbg = None
            if capture_diag:
                Yhat_dbg = pushforward_torch(
                    Xs_t,
                    L_module,
                    tau,
                    mode=flow_mode,
                    residual=residual,
                    K=K,
                    rk=rk,
                    cache=exp_cache,
                    L_override=L_tmp_proj,
                )
                loss_uot_dbg = sinkhorn(Yhat_dbg, Xt_t)
                jac_dbg = 0.0
                if jac_penalty_weight > 0.0 and residual is not None:
                    with torch.enable_grad():
                        sample = Xs_t[:: max(1, Xs_t.shape[0] // 8)][:32]
                        jac_dbg = float((jac_penalty_weight * approx_jacobian_spectral_norm(residual, sample, iters=1)).detach().cpu())
            else:
                jac_dbg = 0.0

        loss_total.backward()

        in_swa = use_swa_phase and step_idx >= swa_start_step
        current_lr = get_swa_lr(step_idx) if in_swa else lr
        for group in optimizer.param_groups:
            group['lr'] = current_lr
        optimizer.step()

        loss_value = float(loss_total.detach().cpu())
        if loss_value < float(best_loss.detach().cpu()) or best_L is None:
            best_loss = loss_total.detach()
            best_L = L_tmp_proj.detach().clone()
            if capture_diag:
                diag.update({
                    'loss_uot': (float(loss_uot_dbg.detach().cpu()) if loss_uot_dbg is not None else loss_value),
                    'loss_total': loss_value,
                    'mode': flow_mode,
                    'K': float(K),
                    'rk': rk,
                    'jac_penalty_weight': float(jac_penalty_weight),
                    'jac_penalty': float(jac_dbg),
                    'meta_active': 1.0 if meta_head is not None else 0.0,
                })
                if residual is not None:
                    diag['residual_scale'] = float(residual.scale.detach().cpu())
                best_diag = diag.copy()

        if in_swa:
            if swa_sum is None:
                swa_sum = L_tmp_proj.detach().clone()
            else:
                swa_sum = swa_sum + L_tmp_proj.detach()
            swa_count += 1

    if use_swa_phase and swa_count > 0 and swa_sum is not None:
        swa_avg = swa_sum / swa_count
        swa_stable = _torch_project_stable(swa_avg, alpha)
        swa_proj, _ = _torch_project_to_stable(swa_stable, stable_margin, soft_weight)
        Yhat_swa = pushforward_torch(
            Xs_t,
            L_module,
            tau,
            mode=flow_mode,
            residual=residual,
            K=K,
            rk=rk,
            cache=exp_cache,
            L_override=swa_proj,
        )
        loss_swa = sinkhorn(Yhat_swa, Xt_t)
        if reg_nuc:
            loss_swa = loss_swa + reg_nuc * torch.linalg.matrix_norm(swa_proj, ord='nuc')
        if use_soft_penalty:
            loss_swa = loss_swa + _torch_soft_stability_penalty(swa_avg, alpha_penalty, lambda_soft)
        if jac_penalty_weight > 0.0 and residual is not None:
            loss_swa = loss_swa + _jacobian_penalty(Xs_t)
        loss_swa_value = float(loss_swa.detach().cpu())
        if loss_swa_value <= float(best_loss.detach().cpu()):
            best_L = swa_proj.detach().clone()
            if capture_diag:
                best_diag = {
                    'loss_uot': float(loss_swa.detach().cpu()),
                    'loss_total': loss_swa_value,
                    'swa_updates': float(swa_count),
                    'mode': flow_mode,
                    'K': float(K),
                    'rk': rk,
                    'jac_penalty_weight': float(jac_penalty_weight),
                    'meta_active': 1.0 if meta_head is not None else 0.0,
                }
                if residual is not None:
                    best_diag['residual_scale'] = float(residual.scale.detach().cpu())

    if best_L is None:
        return None

    L_np = best_L.cpu().numpy()
    L_final, diag_final = project_to_stable(L_np, margin=margin_value, enable=stable_margin is not None)
    if capture_diag:
        final_diag = dict(best_diag or {})
        final_diag.update(diag_final)
    else:
        final_diag = {}
    return L_final, final_diag


__all__ = ["fit_branch_generator", "fit_low_rank_generator"]
