from __future__ import annotations

import torch

__all__ = ["approx_jacobian_spectral_norm"]


def approx_jacobian_spectral_norm(f, x: torch.Tensor, *, iters: int = 1) -> torch.Tensor:
    """Approximate ||J_f||_2 using power iterations."""

    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    v = torch.randn_like(x)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    for _ in range(max(iters, 1)):
        y = f(x)
        (Jv,) = torch.autograd.grad(y, x, v, retain_graph=True, create_graph=True)
        v = Jv.detach()
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    y = f(x)
    (Jv,) = torch.autograd.grad(y, x, v, retain_graph=False, create_graph=False)
    vals = (Jv * v).sum(dim=-1).abs()
    return vals.mean()
