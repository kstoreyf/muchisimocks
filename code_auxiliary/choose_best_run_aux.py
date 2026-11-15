"""Functions moved out of code/choose_best_run.py in the v2 cleanup (2026-11-15).

No longer used by the pipeline; kept for reference / possible reuse, not maintained.
Imports are copied from code/choose_best_run.py; module-level names come from the star import.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from __future__ import annotations
import argparse
import csv
import multiprocessing as mp
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import data_loader
import paths
import sbi_model
from generate_config_inference import (  # noqa: E402
    BX_SWEEP,
    N_TRAIN_SWEEP,
    NOISE_MODES,
    resolve_test_scenario_tags,
    resolve_train_tag_bundle,
    build_sweep_name,
    tags_mask_for_sweep,
    TAG_PARAMS_COVERAGE,
    TAG_SWEEP,
)
from choose_best_run import *  # noqa: F401,F403


def _coverage_batch_unscaled(
    y_cov_nb: list[np.ndarray],
    start: int,
    batch: int,
) -> list[np.ndarray]:
    out = []
    for block in y_cov_nb:
        end = min(start + batch, block.shape[0])
        out.append(np.asarray(block[start:end], dtype=np.float64))
    return out


