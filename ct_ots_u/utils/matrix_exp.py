"""Numerically stable matrix exponential actions."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["expm_action"]


def _pade13(A: Tensor) -> tuple[Tensor, Tensor]:
    b = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ]
    ident = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2

    U = (
        A
        @ (
            A6 * b[13]
            + A4 * b[11]
            + A2 * b[9]
            + ident * b[7]
        )
    )
    U = U + (
        A6 * b[5]
        + A4 * b[3]
        + A2 * b[1]
    )

    V = (
        A6 * b[12]
        + A4 * b[10]
        + A2 * b[8]
        + ident * b[6]
    )
    V = V + (
        A6 * b[4]
        + A4 * b[2]
        + A2 * b[0]
    )

    return U, V


def expm_action(A: Tensor, B: Tensor, t: float = 1.0, *, use_double: bool = True) -> Tensor:
    if use_double and A.dtype != torch.float64:
        A = A.to(torch.float64)
        B = B.to(torch.float64)

    A = t * A
    normA = torch.linalg.norm(A, ord=float("inf"))
    if normA == 0:
        result = B
    else:
        s = max(0, int(torch.ceil(torch.log2(normA / 5.371920351148152)).item()))
        Ascaled = A / (2**s)
        U, V = _pade13(Ascaled)
        P = V + U
        Q = V - U
        X = torch.linalg.solve(Q, P)
        for _ in range(s):
            X = X @ X
        result = B @ X.transpose(-1, -2)

    if use_double and B.dtype != torch.float64:
        result = result.to(B.dtype)
    return result

