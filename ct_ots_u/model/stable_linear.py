"""Stability-preserving linear dynamics modules."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn

from ..utils.matrix_exp import expm_action

__all__ = [
    "DampedSkewLinear",
    "OrthogonalStep",
    "build_dynamics",
]


class DampedSkewLinear(nn.Module):
    """Linear dynamics with antisymmetric structure and learnable damping."""

    def __init__(
        self,
        dim: int,
        *,
        gamma: float = 0.1,
        dt: float = 1.0,
        discrete: bool = True,
    ) -> None:
        super().__init__()
        self.M = nn.Parameter(torch.empty(dim, dim))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.dt = float(dt)
        self.discrete = bool(discrete)

    def A(self) -> Tensor:
        """Return the continuous generator matrix with intrinsic damping."""

        sk = self.M - self.M.transpose(-1, -2)
        dim = sk.shape[-1]
        damping = torch.relu(self.gamma)
        return sk - torch.eye(dim, device=sk.device, dtype=sk.dtype) * damping

    def step(self, z: Tensor, *, dt: Optional[float] = None) -> Tensor:
        """Advance the latent state by one time step."""

        delta_t = float(self.dt if dt is None else dt)
        A = self.A()
        if self.discrete:
            return z + delta_t * (z @ A.transpose(-1, -2))
        return expm_action(A, z, delta_t)


class OrthogonalStep(nn.Module):
    """Linear dynamics based on an orthogonal map followed by contraction."""

    def __init__(self, dim: int, *, rho: float = 0.99, dt: float = 1.0) -> None:
        super().__init__()
        self.B = nn.Parameter(torch.empty(dim, dim))
        nn.init.orthogonal_(self.B)
        self.log_rho = nn.Parameter(torch.log(torch.tensor(float(rho))))
        self.dt = float(dt)

    def _cayley(self) -> Tensor:
        mat = self.B - self.B.transpose(-1, -2)
        eye = torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)
        return torch.linalg.solve(eye + 0.5 * mat, eye - 0.5 * mat)

    def step(self, z: Tensor, *, dt: Optional[float] = None) -> Tensor:
        del dt  # intentionally unused; kept for signature symmetry
        Q = self._cayley()
        rho = torch.sigmoid(self.log_rho)
        return rho * (z @ Q.transpose(-1, -2))


def build_dynamics(
    method: Literal["damped_skew", "orthogonal"],
    *,
    dim: int,
    gamma: float = 0.1,
    rho: float = 0.99,
    dt: float = 1.0,
    discrete: bool = True,
) -> nn.Module:
    """Factory for stability-preserving linear dynamics modules."""

    method = method.lower()
    if method == "damped_skew":
        return DampedSkewLinear(dim, gamma=gamma, dt=dt, discrete=discrete)
    if method == "orthogonal":
        return OrthogonalStep(dim, rho=rho, dt=dt)
    raise ValueError(f"Unsupported dynamics method '{method}'")

