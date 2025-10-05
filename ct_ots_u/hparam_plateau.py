"""Plateau detection for tau/reg grid (self-R experiment)."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np


def find_plateau(stats: Dict[Tuple[float, float], Dict[str, float]]) -> Dict[str, object]:
    """Given {(tau, reg): {"mean": m, "se": s}}, find 1-SE plateau cluster."""
    if not stats:
        return {
            "threshold": None,
            "plateau_entries": [],
            "component_size": 0,
            "taus": [],
            "regs": [],
            "grid": [],
        }

    taus = sorted({k[0] for k in stats.keys()})
    regs = sorted({k[1] for k in stats.keys()})

    items = list(stats.items())
    means = np.array([v["mean"] for _, v in items])
    ses = np.array([v["se"] for _, v in items])
    best_idx = int(np.argmin(means))
    threshold = float(means[best_idx] + ses[best_idx])

    grid = np.zeros((len(taus), len(regs)), dtype=bool)
    for (tau, reg), val in stats.items():
        i = taus.index(tau)
        j = regs.index(reg)
        grid[i, j] = val["mean"] <= threshold

    component = _largest_component(grid)
    plateau_entries: List[Dict[str, float]] = []
    for i, j in component:
        tau = taus[i]
        reg = regs[j]
        entry = stats[(tau, reg)]
        plateau_entries.append(
            {
                "tau": float(tau),
                "reg": float(reg),
                "mean": entry["mean"],
                "se": entry["se"],
            }
        )

    return {
        "threshold": threshold,
        "plateau_entries": plateau_entries,
        "component_size": len(component),
        "taus": taus,
        "regs": regs,
        "grid": grid.tolist(),
    }


def _largest_component(mask: np.ndarray) -> List[Tuple[int, int]]:
    visited = np.zeros_like(mask, dtype=bool)
    best: List[Tuple[int, int]] = []
    rows, cols = mask.shape

    for i in range(rows):
        for j in range(cols):
            if not mask[i, j] or visited[i, j]:
                continue
            comp = []
            queue = deque([(i, j)])
            visited[i, j] = True
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            if len(comp) > len(best):
                best = comp
    return best
