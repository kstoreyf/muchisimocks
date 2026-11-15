"""Functions moved out of code/utils_plot.py in the v2 cleanup (2026-11-15).

No longer used by the pipeline; kept for reference / possible reuse, not maintained.
Imports are copied from code/utils_plot.py; module-level names come from the star import.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Union
import numpy as np
from pathlib import Path
import data_loader
import paths
import utils_inference
from utils_plot import *  # noqa: F401,F403


def _sbi_samples_pred_path(tag_inf: str, tag_test: str) -> Path:
    """Path to finalized test predictions (`samples_test*_pred.npy`) that loaders use."""
    dir_sbi = paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_inf}"
    return dir_sbi / f"samples_test{tag_test}_pred.npy"


def sbi_samples_exist(tag_inf: str, tag_test: str) -> bool:
    """Whether finalized SBI posterior samples for this tag_test exist on disk."""
    return _sbi_samples_pred_path(tag_inf, tag_test).exists()


def filter_available_chains(
    tags_inf: Sequence[str],
    tags_test: Sequence[str],
    inf_methods: Sequence[str],
    labels: Sequence[str],
    colors: Sequence[str],
):
    """Filter to only those (tag_inf, tag_test) pairs that exist on disk."""
    keep = [sbi_samples_exist(ti, tt) for ti, tt in zip(tags_inf, tags_test)]
    if not any(keep):
        return [], [], [], [], []
    tags_inf_kept = [t for t, k in zip(tags_inf, keep) if k]
    tags_test_kept = [t for t, k in zip(tags_test, keep) if k]
    inf_methods_kept = [m for m, k in zip(inf_methods, keep) if k]
    labels_kept = [l for l, k in zip(labels, keep) if k]
    colors_kept = [c for c, k in zip(colors, keep) if k]
    return tags_inf_kept, tags_test_kept, inf_methods_kept, labels_kept, colors_kept


