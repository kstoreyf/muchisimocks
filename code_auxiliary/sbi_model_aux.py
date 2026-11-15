"""Functions moved out of code/sbi_model.py in the v2 cleanup (2026-11-15).

No longer used by the pipeline; kept for reference / possible reuse, not maintained.
Imports are copied from code/sbi_model.py; module-level names come from the star import.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import multiprocessing as mp
import os
import queue
import re
import shutil
from types import SimpleNamespace
import numpy as np
import pathlib
import math
import pickle
import time
import tempfile
import yaml
import wandb
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
import torch
from torch.distributions import biject_to, constraints
from sbi.utils import BoxUniform
from sbi.utils.sbiutils import within_support
import paths
import scaler_custom as scl
import utils_inference
from sbi_model import *  # noqa: F401,F403


def wandb_sweep_best_run(sweep, *, minimize: str):
    """
    ``wandb.apis.public.Sweep.best_run`` only accepts ``order=``, not ``minimize=``.
    Minimizing a summary metric is ``order='+<name>'`` (see ``QueryGenerator``).

    Raises ``ValueError`` if the selected run has no ``minimize`` value in summary
    (metric never logged); do not treat an unlogged metric as a comparable best.
    """
    run = sweep.best_run(order=f"+{minimize}")
    if run is None:
        raise ValueError(
            "No run returned for sweep id=%r (empty sweep or filter)."
            % (getattr(sweep, "id", None),)
        )
    sm = dict(run.summary)
    val = sm.get(minimize)
    if val is None:
        raise ValueError(
            "Best run %s has no summary[%r] logged; cannot select best without that metric."
            % (run.id, minimize)
        )
    return run


