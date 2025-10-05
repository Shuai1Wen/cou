"""Statistical utilities (bootstrap CI, paired tests, JSON dumping)."""

from __future__ import annotations

from ..stats_eval import bootstrap_ci, dump_json, paired_test

__all__ = ["bootstrap_ci", "dump_json", "paired_test"]
