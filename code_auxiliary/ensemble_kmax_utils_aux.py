"""Functions moved out of code/ensemble_kmax_utils.py in the v2 cleanup (2026-11-15).

No longer used by the pipeline; kept for reference / possible reuse, not maintained.
Imports are copied from code/ensemble_kmax_utils.py; module-level names come from the star import.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import paths
import utils_plot
from choose_best_run import build_sweep_tag_inf, sweep_root_for_tag_inf
from generate_config_inference import (
    build_tag_data,
    resolve_test_scenario_tags,
    resolve_train_tag_bundle,
    tags_mask_for_sweep,
    TAG_PARAMS_COVERAGE,
    TAG_PARAMS_SHAME,
    TAG_SWEEP,
)
from ensemble_kmax_utils import *  # noqa: F401,F403


def sweep_tag_inf_for_statistics(statistics: list[str]) -> str:
    train = _train_bundle()
    return build_sweep_tag_inf(
        data_mode=DATA_MODE,
        statistics=statistics,
        tags_mask=tags_mask_for_sweep(statistics),
        tag_params=train["tag_params"],
        tag_biasparams=train["tag_biasparams"],
        tag_noise=train["tag_noise"],
        reparameterize=True,
        tag_sweep=TAG_SWEEP,
    )


def sweep_root_for_statistics(statistics: list[str]) -> Path:
    return sweep_root_for_tag_inf(sweep_tag_inf_for_statistics(statistics))


