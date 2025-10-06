"""Alignment losses that avoid optimal-transport solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "AlignLoss",
    "SWDLoss",
    "MMDLoss",
    "CORALLoss",
    "DANNLoss",
    "build_aligner",
]


def _whiten(z: Tensor, eps: float = 1e-5) -> Tensor:
    mean = z.mean(dim=0, keepdim=True)
    centered = z - mean
    std = centered.pow(2).mean(dim=0, keepdim=True).add(eps).sqrt()
    return centered / std


class AlignLoss(nn.Module):
    def forward(self, z_s: Tensor, z_t: Tensor) -> Tensor:  # pragma: no cover - interface only
        raise NotImplementedError


class SWDLoss(AlignLoss):
    def __init__(
        self,
        num_proj: int = 128,
        p: int = 2,
        *,
        orthogonal: bool = True,
    ) -> None:
        super().__init__()
        self.num_proj = int(num_proj)
        self.p = int(p)
        self.orthogonal = bool(orthogonal)

    def _sample_dirs(self, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        dirs = torch.randn(dim, self.num_proj, device=device, dtype=dtype)
        dirs = torch.nn.functional.normalize(dirs, dim=0)
        if self.orthogonal and self.num_proj <= dim:
            q, _ = torch.linalg.qr(dirs, mode="reduced")
            dirs = q[:, : self.num_proj]
        return dirs

    def forward(self, z_s: Tensor, z_t: Tensor) -> Tensor:
        z_s = _whiten(z_s)
        z_t = _whiten(z_t)
        dirs = self._sample_dirs(z_s.shape[1], z_s.device, z_s.dtype)
        proj_s = z_s @ dirs
        proj_t = z_t @ dirs
        proj_s, _ = proj_s.sort(dim=0)
        proj_t, _ = proj_t.sort(dim=0)
        diff = (proj_s - proj_t).abs().pow(self.p)
        return diff.mean()


class MMDLoss(AlignLoss):
    def __init__(self, bandwidths: Sequence[float] = (0.5, 1.0, 2.0)) -> None:
        super().__init__()
        self.bandwidths = tuple(float(b) for b in bandwidths)

    def _gaussian(self, x: Tensor, y: Tensor, sigma: float) -> Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist2 = diff.pow(2).sum(dim=-1)
        return torch.exp(-dist2 / (2.0 * sigma**2))

    def forward(self, z_s: Tensor, z_t: Tensor) -> Tensor:
        z_s = _whiten(z_s)
        z_t = _whiten(z_t)
        mmd = z_s.new_tensor(0.0)
        for sigma in self.bandwidths:
            K_ss = self._gaussian(z_s, z_s, sigma)
            K_tt = self._gaussian(z_t, z_t, sigma)
            K_st = self._gaussian(z_s, z_t, sigma)
            mmd = mmd + (
                K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean()
            )
        return mmd / len(self.bandwidths)


class CORALLoss(AlignLoss):
    def __init__(self, shrinkage: float = 1e-3) -> None:
        super().__init__()
        self.shrinkage = float(shrinkage)

    def _cov(self, x: Tensor) -> Tensor:
        x = _whiten(x)
        n = x.shape[0]
        c = (x.T @ x) / max(n - 1, 1)
        eye = torch.eye(c.shape[0], device=c.device, dtype=c.dtype)
        return (1 - self.shrinkage) * c + self.shrinkage * eye

    def forward(self, z_s: Tensor, z_t: Tensor) -> Tensor:
        cov_s = self._cov(z_s)
        cov_t = self._cov(z_t)
        diff = cov_s - cov_t
        return diff.pow(2).mean()


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:  # pragma: no cover - simple autograd op
        ctx.save_for_backward(torch.tensor(lambd, device=x.device, dtype=x.dtype))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:  # pragma: no cover
        (lambd,) = ctx.saved_tensors
        return -lambd * grad_output, None


class _DomainDiscriminator(nn.Module):
    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class DANNLoss(AlignLoss):
    def __init__(self, dim: int, hidden: int = 128, lambd: float = 1.0) -> None:
        super().__init__()
        self.disc = _DomainDiscriminator(dim, hidden)
        self.lambd = float(lambd)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, z_s: Tensor, z_t: Tensor) -> Tensor:
        z_s = _GradReverse.apply(_whiten(z_s), self.lambd)
        z_t = _GradReverse.apply(_whiten(z_t), self.lambd)
        logits_s = self.disc(z_s)
        logits_t = self.disc(z_t)
        labels_s = torch.zeros_like(logits_s)
        labels_t = torch.ones_like(logits_t)
        loss = self.bce(logits_s, labels_s) + self.bce(logits_t, labels_t)
        return loss * 0.5


@dataclass
class AlignConfig:
    method: Literal["none", "swd", "mmd", "coral", "dann"] = "none"
    weight: float = 0.0
    num_projections: int = 128
    sliced_p: int = 2
    orthogonal_projections: bool = True
    mmd_bandwidths: Sequence[float] = (0.5, 1.0, 2.0)
    coral_shrinkage: float = 1e-3
    dann_hidden: int = 128
    dann_lambda: float = 1.0


def build_aligner(config: AlignConfig, dim: int) -> AlignLoss | None:
    method = config.method.lower()
    if method == "none" or config.weight <= 0:
        return None
    if method == "swd":
        return SWDLoss(
            num_proj=config.num_projections,
            p=config.sliced_p,
            orthogonal=config.orthogonal_projections,
        )
    if method == "mmd":
        return MMDLoss(config.mmd_bandwidths)
    if method == "coral":
        return CORALLoss(config.coral_shrinkage)
    if method == "dann":
        return DANNLoss(dim, hidden=config.dann_hidden, lambd=config.dann_lambda)
    raise ValueError(f"Unsupported align method '{config.method}'")

