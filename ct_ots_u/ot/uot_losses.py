# -*- coding: utf-8 -*-
"""Unified balanced and unbalanced Sinkhorn losses with configurable backends."""

from __future__ import annotations

from typing import Literal, Tuple

import inspect
import warnings

import numpy as np

try:
    import torch
    from geomloss import SamplesLoss
except ImportError:  # pragma: no cover - optional dependency
    SamplesLoss = None  # type: ignore
    torch = None  # type: ignore

# Optional KeOps acceleration detection
try:  # pragma: no cover - optional dependency
    import pykeops  # noqa: F401
    _HAS_KEOPS = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_KEOPS = False

import ot

Array = np.ndarray

if SamplesLoss is not None:
    _SAMPLESLOSS_PARAMETERS = None
    for _candidate in (
        SamplesLoss,
        getattr(SamplesLoss, '__init__', None),
        getattr(SamplesLoss, '__call__', None),
    ):
        if _candidate is None:
            continue
        try:
            _SAMPLESLOSS_PARAMETERS = inspect.signature(_candidate).parameters
            break
        except (TypeError, ValueError):  # pragma: no cover - signature unavailable
            continue
    if _SAMPLESLOSS_PARAMETERS is None:
        _SAMPLESLOSS_PARAMETERS = {}
else:  # pragma: no cover - optional dependency path
    _SAMPLESLOSS_PARAMETERS = {}


def _kw_supported(name: str) -> bool:
    if not _SAMPLESLOSS_PARAMETERS:
        return False
    if name in _SAMPLESLOSS_PARAMETERS:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in _SAMPLESLOSS_PARAMETERS.values())


def _make_samples_loss(
    *,
    blur: float,
    debias: bool = True,
    backend: str | None = None,
    scaling: float | None = None,
    reach: float | None = None,
):
    if SamplesLoss is None:  # pragma: no cover - optional dependency
        raise ImportError('geomloss is required but unavailable')
    kwargs = {'blur': blur, 'debias': debias}
    if backend is not None and _kw_supported('backend'):
        kwargs['backend'] = backend
    if scaling is not None and _kw_supported('scaling'):
        kwargs['scaling'] = scaling
    if reach is not None and _kw_supported('reach'):
        kwargs['reach'] = reach
    return SamplesLoss('sinkhorn', **kwargs)


def _geomloss_available() -> bool:
    return SamplesLoss is not None and torch is not None


def _geomloss_candidates(backend: str | None, scaling: float | None) -> list[tuple[str | None, float | None]]:
    attempts: list[tuple[str | None, float | None]] = []
    seen: set[tuple[str | None, float | None]] = set()

    def _push(b: str | None, s: float | None) -> None:
        key = (b, s)
        if key in seen:
            return
        seen.add(key)
        attempts.append(key)

    # Prefer KeOps if available
    if _HAS_KEOPS:
        _push("keops", scaling)

    # Then the explicitly requested backend
    _push(backend, scaling)

    # Broaden search for robust fallbacks
    for candidate in ("online", "auto", "multiscale", "tensorized"):
        _push(candidate, scaling)
    _push(None, scaling)
    _push(None, None)
    return attempts


def _evaluate_geomloss(
    xt,
    yt,
    *,
    blur: float,
    reach: float | None,
    backend: str,
    scaling: float,
):
    attempts = _geomloss_candidates(backend, scaling)

    last_exc: Exception | None = None
    for candidate_backend, candidate_scaling in attempts:
        try:
            loss = _make_samples_loss(
                blur=blur,
                debias=True,
                backend=candidate_backend,
                scaling=candidate_scaling,
                reach=reach,
            )
            value = loss(xt, yt)
            return float(value.detach().cpu().item())
        except Exception as exc:  # pragma: no cover - runtime dependent
            last_exc = exc
            continue

    if last_exc is None:  # pragma: no cover - defensive
        raise RuntimeError('GeomLoss evaluation failed without exception')
    raise last_exc


def sinkhorn_divergence(
    x: Array,
    y: Array,
    blur: float = 0.05,
    *,
    backend: Literal['geomloss', 'pot'] = 'geomloss',
    sinkhorn_backend: str = 'online',
    scaling: float = 0.9,
    num_iter_max: int = 1000,
    stop_thr: float = 1e-6,
) -> float:
    """Compute debiased Sinkhorn divergence with optional stabilised backend."""

    if backend == 'geomloss':
        if not _geomloss_available():  # pragma: no cover - optional dependency path
            raise ImportError(
                "geomloss is required for backend='geomloss'. Install geomloss or use backend='pot'."
            )
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=device)
        try:
            return _evaluate_geomloss(
                xt,
                yt,
                blur=blur,
                reach=None,
                backend=sinkhorn_backend,
                scaling=scaling,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback
            warnings.warn(
                f"GeomLoss backend unavailable; falling back to POT Sinkhorn divergence ({exc!r})",
                RuntimeWarning,
            )

    # POT fallback with stabilised solver
    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(y.shape[0]) / y.shape[0]
    M_xy = ot.dist(x, y, metric='sqeuclidean')
    scale = float(M_xy.max())
    if scale > 0:
        M_xy = M_xy / scale

    try:
        G_xy = ot.sinkhorn_stabilized(a, b, M_xy, reg=blur, numItermax=num_iter_max, stopThr=stop_thr)
    except AttributeError:  # pragma: no cover - compatibility
        G_xy = ot.sinkhorn(a, b, M_xy, reg=blur, numItermax=num_iter_max, stopThr=stop_thr)
    cost_xy = float(np.sum(G_xy * M_xy))

    M_xx = ot.dist(x, x, metric='sqeuclidean')
    M_yy = ot.dist(y, y, metric='sqeuclidean')
    if M_xx.max() > 0:
        M_xx = M_xx / M_xx.max()
    if M_yy.max() > 0:
        M_yy = M_yy / M_yy.max()

    try:
        G_xx = ot.sinkhorn_stabilized(a, a, M_xx, reg=blur, numItermax=num_iter_max, stopThr=stop_thr)
        G_yy = ot.sinkhorn_stabilized(b, b, M_yy, reg=blur, numItermax=num_iter_max, stopThr=stop_thr)
    except AttributeError:  # pragma: no cover
        G_xx = ot.sinkhorn(a, a, M_xx, reg=blur, numItermax=num_iter_max, stopThr=stop_thr)
        G_yy = ot.sinkhorn(b, b, M_yy, reg=blur, numItermax=num_iter_max, stopThr=stop_thr)

    cost_xx = float(np.sum(G_xx * M_xx))
    cost_yy = float(np.sum(G_yy * M_yy))
    return cost_xy - 0.5 * (cost_xx + cost_yy)


def uot_sinkhorn_cost(
    x: Array,
    y: Array,
    *,
    reg: float = 0.08,
    reg_m: float = 1.0,
    metric: Literal['euclidean', 'sqeuclidean'] = 'euclidean',
    num_iter_max: int = 1000,
    stop_thr: float = 1e-6,
    backend: Literal['geomloss', 'pot'] = 'geomloss',
    sinkhorn_backend: str = 'online',
    scaling: float = 0.9,
    warm_start: tuple[np.ndarray, np.ndarray] | None = None,
    return_warm_start: bool = False,
) -> tuple[float, Array | None, Array | None] | tuple[float, Array | None, Array | None, tuple[np.ndarray | None, np.ndarray | None]]:
    """Unbalanced Sinkhorn transport cost using configurable backend."""

    if backend == 'geomloss':
        if not _geomloss_available():  # pragma: no cover
            raise ImportError(
                "geomloss is required for backend='geomloss'. Install geomloss or use backend='pot'."
            )
        if not _kw_supported('reach'):
            warnings.warn(
                "GeomLoss SamplesLoss missing 'reach'; falling back to POT Sinkhorn",
                RuntimeWarning,
            )
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            xt = torch.as_tensor(x, dtype=torch.float32, device=device)
            yt = torch.as_tensor(y, dtype=torch.float32, device=device)
            try:
                loss = _make_samples_loss(
                    blur=reg,
                    debias=True,
                    backend=sinkhorn_backend,
                    scaling=scaling,
                    reach=reg_m,
                )
                value = loss(xt, yt)
                result = float(value.detach().cpu().item())
                if return_warm_start:
                    return result, None, None, (None, None)
                return result, None, None
            except Exception as exc:  # pragma: no cover - runtime fallback
                warnings.warn(
                    f"GeomLoss backend unavailable; falling back to POT Sinkhorn ({exc!r})",
                    RuntimeWarning,
                )

    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(y.shape[0]) / y.shape[0]
    M = ot.dist(x, y, metric=metric)
    scale = float(M.max())
    if scale > 0:
        M = M / scale

    warm = warm_start if warm_start is not None else None
    log_enabled = bool(return_warm_start)

    try:
        if log_enabled:
            G, log = ot.unbalanced.sinkhorn_unbalanced(
                a,
                b,
                M,
                reg=reg,
                reg_m=reg_m,
                warmstart=warm,
                numItermax=num_iter_max,
                stopThr=stop_thr,
                method='sinkhorn_stabilized',
                verbose=False,
                log=True,
            )
            if log:
                logu, logv = log.get('logu'), log.get('logv')
                next_warm = (logu, logv) if logu is not None and logv is not None else None
            else:
                next_warm = None
        else:
            G = ot.unbalanced.sinkhorn_unbalanced(
                a,
                b,
                M,
                reg=reg,
                reg_m=reg_m,
                warmstart=warm,
                numItermax=num_iter_max,
                stopThr=stop_thr,
                method='sinkhorn_stabilized',
                verbose=False,
            )
            next_warm = None
    except TypeError:  # pragma: no cover - POT versions without method keyword
        if log_enabled:
            G, log = ot.unbalanced.sinkhorn_unbalanced(
                a,
                b,
                M,
                reg=reg,
                reg_m=reg_m,
                warmstart=warm,
                numItermax=num_iter_max,
                stopThr=stop_thr,
                verbose=False,
                log=True,
            )
            if log:
                logu, logv = log.get('logu'), log.get('logv')
                next_warm = (logu, logv) if logu is not None and logv is not None else None
            else:
                next_warm = None
        else:
            G = ot.unbalanced.sinkhorn_unbalanced(
                a,
                b,
                M,
                reg=reg,
                reg_m=reg_m,
                warmstart=warm,
                numItermax=num_iter_max,
                stopThr=stop_thr,
                verbose=False,
            )
            next_warm = None
    cost = float(np.sum(G * M))
    if return_warm_start:
        if next_warm is None:
            next_warm = (None, None)
        return cost, G, M, next_warm
    return cost, G, M


__all__ = ['sinkhorn_divergence', 'uot_sinkhorn_cost']
