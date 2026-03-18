"""Central configuration for muchisimocks filesystem paths.

The idea is:
- Change base paths (scratch, shared storage) here or via environment variables
- Keep all hard-coded '/scratch/kstoreyf/muchisimocks/...' in one place

Env vars (optional, with current defaults in parentheses):
- MUCHISIMOCKS_SCRATCH    (/scratch/kstoreyf/muchisimocks)
- MUCHISIMOCKS_RESULTS    (<SCRATCH>/results)
- MUCHISIMOCKS_DATA       (<SCRATCH>/data)
- MUCHISIMOCKS_LIB        (<SCRATCH>/muchisimocks_lib)
- MUCHISIMOCKS_LIB_OOD    (<SCRATCH>/muchisimocks_lib_ood)
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Base scratch directory for this project
SCRATCH_ROOT = Path(
    os.environ.get("MUCHISIMOCKS_SCRATCH", "/scratch/kstoreyf/muchisimocks")
)

# Common subdirectories under scratch
DIR_RESULTS = Path(
    os.environ.get("MUCHISIMOCKS_RESULTS", SCRATCH_ROOT / "results")
)
DIR_DATA = Path(
    os.environ.get("MUCHISIMOCKS_DATA", SCRATCH_ROOT / "data")
)
DIR_LIB = Path(
    os.environ.get("MUCHISIMOCKS_LIB", SCRATCH_ROOT / "muchisimocks_lib")
)
DIR_LIB_OOD = Path(
    os.environ.get("MUCHISIMOCKS_LIB_OOD", SCRATCH_ROOT / "muchisimocks_lib_ood")
)


def mocks_lib_dir(tag_params: str) -> Path:
    """Directory containing main muchisimocks library for a given tag_params."""
    return DIR_LIB.with_name(f"{DIR_LIB.name}{tag_params}")


def mocks_lib_ood_dir(subdir: str = "shame") -> Path:
    """Directory for OOD libraries (e.g. 'shame')."""
    return DIR_LIB_OOD / subdir


def noise_fields_dir(tag_noise: str) -> Path:
    """Directory where noise fields are stored for a given tag_noise."""
    return DIR_DATA / "noise_fields" / f"fields{tag_noise}"


def statistics_dir(statistic: str, tag_mocks: str) -> Path:
    """Directory where precomputed statistics (pk, bispec, pgm, pnn, ...) are stored."""
    return DIR_DATA / f"{statistic}s_mlib" / f"{statistic}s{tag_mocks}"

