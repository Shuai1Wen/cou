"""Training utilities using structurally stable dynamics and OT-free alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import AlignCfg, DynamicsCfg, RegularCfg
from .alignment import AlignConfig, build_aligner
from .stable_linear import DampedSkewLinear, OrthogonalStep, build_dynamics

Array = np.ndarray

__all__ = ["fit_branch_generator", "fit_low_rank_generator"]


@dataclass
class _TrainDiagnostics:
    total_loss: float
    task_loss: float
    align_loss: float
    lyapunov: float
    lipschitz: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss_total": self.total_loss,
            "loss_task": self.task_loss,
            "loss_align": self.align_loss,
            "lyapunov": self.lyapunov,
            "lipschitz": self.lipschitz,
        }


def _lyapunov_penalty(A: Optional[Tensor], alpha: float) -> Tensor:
    if A is None:
        return torch.zeros((), device=torch.device("cpu"))
    sym = 0.5 * (A + A.transpose(-1, -2))
    shifted = sym + alpha * torch.eye(sym.shape[0], device=sym.device, dtype=sym.dtype)
    eig_max = torch.linalg.eigvalsh(shifted).max()
    return torch.clamp(eig_max, min=0.0)


def _module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _lipschitz_penalty(modules: list[nn.Module], target: float, weight: float) -> Tensor:
    if weight <= 0:
        device = _module_device(modules[0])
        return torch.zeros((), device=device)
    log_norm = None
    for module in modules:
        for param in module.parameters():
            if param.dim() >= 2:
                sigma = torch.linalg.matrix_norm(param, ord=2)
                if log_norm is None:
                    log_norm = torch.log(torch.clamp(sigma, min=1e-6))
                else:
                    log_norm = log_norm + torch.log(torch.clamp(sigma, min=1e-6))
    if log_norm is None:
        device = _module_device(modules[0])
        return torch.zeros((), device=device)
    lipschitz = torch.exp(log_norm)
    return weight * torch.clamp(lipschitz - target, min=0.0)


def _export_matrix(dynamics: nn.Module) -> Tensor:
    if isinstance(dynamics, DampedSkewLinear):
        return dynamics.A()
    if isinstance(dynamics, OrthogonalStep):
        Q = dynamics._cayley()  # type: ignore[attr-defined]
        rho = torch.sigmoid(dynamics.log_rho)
        return rho * Q
    raise TypeError(f"Unsupported dynamics module {type(dynamics)!r}")


def _train_dynamics(
    Xs: Array,
    Xt: Array,
    *,
    steps: int,
    lr: float,
    device: str,
    dynamics_cfg: DynamicsCfg,
    regular_cfg: RegularCfg,
    align_cfg: AlignCfg,
) -> tuple[np.ndarray, Dict[str, float]]:
    if steps <= 0:
        raise ValueError("steps must be positive")

    torch_device = torch.device(device)
    dtype = torch.float32
    src = torch.as_tensor(Xs, device=torch_device, dtype=dtype)
    tgt = torch.as_tensor(Xt, device=torch_device, dtype=dtype)

    dynamics = build_dynamics(
        dynamics_cfg.backend,
        dim=src.shape[1],
        gamma=dynamics_cfg.gamma,
        rho=dynamics_cfg.rho,
        dt=dynamics_cfg.dt,
        discrete=dynamics_cfg.discrete,
    ).to(torch_device, dtype=dtype)

    align_config = AlignConfig(
        method=align_cfg.method,
        weight=align_cfg.weight,
        num_projections=align_cfg.num_projections,
        sliced_p=align_cfg.sliced_p,
        orthogonal_projections=align_cfg.orthogonal_projections,
        mmd_bandwidths=align_cfg.mmd_bandwidths,
        coral_shrinkage=align_cfg.coral_shrinkage,
        dann_hidden=align_cfg.dann_hidden,
        dann_lambda=align_cfg.dann_lambda,
    )
    aligner = build_aligner(align_config, src.shape[1])
    if aligner is not None:
        aligner = aligner.to(torch_device)

    params = list(dynamics.parameters())
    if aligner is not None:
        params += [p for p in aligner.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=lr, amsgrad=True)

    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in dynamics.state_dict().items()}
    best_diag = _TrainDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0)

    for _ in range(steps):
        optimizer.zero_grad()

        pred = dynamics.step(src)
        task_loss = F.mse_loss(pred, tgt)

        if aligner is not None:
            align_loss = aligner(pred, tgt)
            total_loss = task_loss + align_cfg.weight * align_loss
        else:
            align_loss = torch.zeros((), device=torch_device, dtype=dtype)
            total_loss = task_loss

        lyap_loss = torch.zeros_like(total_loss)
        if isinstance(dynamics, DampedSkewLinear):
            lyap_loss = _lyapunov_penalty(dynamics.A(), regular_cfg.lyapunov_alpha)
            total_loss = total_loss + regular_cfg.lyapunov_lambda * lyap_loss

        lip_loss = torch.zeros_like(total_loss)
        if regular_cfg.lipschitz_enable:
            lip_loss = _lipschitz_penalty([dynamics], regular_cfg.lipschitz_target, regular_cfg.lipschitz_lambda)
            total_loss = total_loss + lip_loss

        total_loss.backward()
        optimizer.step()

        current = total_loss.detach().item()
        if current < best_loss:
            best_loss = current
            best_state = {k: v.detach().clone() for k, v in dynamics.state_dict().items()}
            best_diag = _TrainDiagnostics(
                total_loss=total_loss.detach().item(),
                task_loss=task_loss.detach().item(),
                align_loss=align_loss.detach().item(),
                lyapunov=lyap_loss.detach().item(),
                lipschitz=lip_loss.detach().item(),
            )

    dynamics.load_state_dict(best_state)
    matrix = _export_matrix(dynamics).detach().cpu().numpy()
    return matrix, best_diag.to_dict()


def fit_branch_generator(
    Xs: Array,
    Xt: Array,
    *,
    tau: float = 0.5,
    rank: int | None = None,
    steps: int = 200,
    lr: float = 1e-3,
    alpha: float = 1e-3,
    reg_nuc: float = 0.0,
    seed: int = 0,
    use_torch: bool | None = None,
    use_torch_compile: bool | None = None,
    device: str = "cpu",
    reg: float = 0.0,
    reg_m: float = 0.0,
    num_iter_max: int = 0,
    stop_thr: float = 0.0,
    stable_margin: float | None = None,
    soft_weight: float | None = None,
    use_soft_penalty: bool | None = None,
    lambda_soft: float = 0.0,
    use_swa: bool = False,
    swa_start_ratio: float = 0.0,
    swa_lr: float = 0.0,
    swa_schedule: str = "constant",
    log_diagnostics: bool | None = False,
    backend: str = "geomloss",
    sinkhorn_backend: str | None = None,
    sinkhorn_scaling: float = 0.0,
    sinkhorn_minibatch: bool = False,
    sinkhorn_batch_size: int | None = None,
    model: str = "linear",
    K: int = 1,
    residual_cfg: dict[str, Any] | None = None,
    meta_cfg: dict[str, Any] | None = None,
    rk: str = "rk2",
    jac_penalty_weight: float = 0.0,
    dynamics_cfg: Optional[DynamicsCfg] = None,
    regular_cfg: Optional[RegularCfg] = None,
    align_cfg: Optional[AlignCfg] = None,
) -> Array | tuple[Array, Dict[str, float]]:
    del (
        tau,
        rank,
        alpha,
        reg_nuc,
        seed,
        use_torch,
        use_torch_compile,
        reg,
        reg_m,
        num_iter_max,
        stop_thr,
        stable_margin,
        soft_weight,
        use_soft_penalty,
        lambda_soft,
        use_swa,
        swa_start_ratio,
        swa_lr,
        swa_schedule,
        backend,
        sinkhorn_backend,
        sinkhorn_scaling,
        sinkhorn_minibatch,
        sinkhorn_batch_size,
        model,
        K,
        residual_cfg,
        meta_cfg,
        rk,
        jac_penalty_weight,
    )

    dynamics_cfg = dynamics_cfg or DynamicsCfg()
    regular_cfg = regular_cfg or RegularCfg()
    align_cfg = align_cfg or AlignCfg()

    matrix, diagnostics = _train_dynamics(
        Xs,
        Xt,
        steps=steps,
        lr=lr,
        device=device,
        dynamics_cfg=dynamics_cfg,
        regular_cfg=regular_cfg,
        align_cfg=align_cfg,
    )

    if log_diagnostics:
        return matrix, diagnostics
    return matrix


def fit_low_rank_generator(
    Xs: Array,
    Xt: Array,
    rank: int,
    *,
    tau: float = 0.5,
    steps: int = 200,
    lr: float = 1e-3,
    alpha: float = 1e-3,
    seed: int = 0,
    reg: float = 0.0,
    reg_m: float = 0.0,
    num_iter_max: int = 0,
    stop_thr: float = 0.0,
    stable_margin: float | None = None,
    soft_weight: float | None = None,
    backend: str = "geomloss",
    sinkhorn_backend: str | None = None,
    sinkhorn_scaling: float = 0.0,
    dynamics_cfg: Optional[DynamicsCfg] = None,
    regular_cfg: Optional[RegularCfg] = None,
    align_cfg: Optional[AlignCfg] = None,
) -> tuple[Array, Array, Array]:
    del (
        tau,
        alpha,
        seed,
        reg,
        reg_m,
        num_iter_max,
        stop_thr,
        stable_margin,
        soft_weight,
        backend,
        sinkhorn_backend,
        sinkhorn_scaling,
    )

    dynamics_cfg = dynamics_cfg or DynamicsCfg()
    regular_cfg = regular_cfg or RegularCfg()
    align_cfg = align_cfg or AlignCfg()

    matrix, _ = _train_dynamics(
        Xs,
        Xt,
        steps=steps,
        lr=lr,
        device="cpu",
        dynamics_cfg=dynamics_cfg,
        regular_cfg=regular_cfg,
        align_cfg=align_cfg,
    )

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vt[:rank].T
    return U_r, V_r, matrix

