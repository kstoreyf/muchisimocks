"""Utilities for K-member ensemble inference across k_max scale cuts.

Hyperparameters come from the fiducial rand30 sweep for each statistic combo
(``tags_mask_for_sweep``). When no sweep exists at a given k cut, the same
top-N hyperparameters are retrained at that mask (``nth_best_run`` from the
fiducial sweep).

Ensemble member 0 uses the existing ``_best-rand30`` directory when present;
members 1..K-1 use ``_best-rand30_nbest{n}``.
"""

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
)

TAG_SWEEP = "-rand30"
BX = 32
N_TRAIN = 10000
NOISE_MODE = "noisy"
TAG_PARAMS_TRAIN = "_p5_n10000"
DATA_MODE = "muchisimocks"
TAG_REPARAM = "_rp"
KMAX_PK_LOOSE = 0.4

TAGS_KMAX_KB = ["_kb0.1", "_kb0.15", "_kb0.2", "_kb0.25", "_kb0.3", "_kb0.35", ""]
TAGS_KMAX_KPGM = [
    "_kpgm0.1",
    "_kpgm0.15",
    "_kpgm0.2",
    "_kpgm0.25",
    "_kpgm0.3",
    "_kpgm0.35",
    "",
]
K_OVERALL = [0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.35, 0.37, 0.4]
STAT_ROWS = [["pk"], ["pk", "pgm"], ["pk", "bispec"], ["pk", "bispec", "pgm"]]
NBAR_TAGS_SHAME = ["_nbar0.00011", "_nbar0.00022", "_nbar0.00054"]
N_ENSEMBLE_K = 3


def _train_bundle():
    return resolve_train_tag_bundle(TAG_PARAMS_TRAIN, NOISE_MODE)


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


def parse_mask_to_tags_mask(statistics: list[str], mask: str) -> list[str]:
    tags_mask: list[str] = []
    for stat in statistics:
        if stat == "pk":
            m = re.search(r"_kp[\d.]+", mask)
            tags_mask.append(m.group(0) if m else "")
        elif stat == "bispec":
            m = re.search(r"_kb[\d.]+", mask)
            tags_mask.append(m.group(0) if m else "")
        elif stat == "pgm":
            m = re.search(r"_kpgm[\d.]+", mask)
            tags_mask.append(m.group(0) if m else "")
        else:
            raise ValueError(stat)
    if len(tags_mask) != len(statistics):
        raise ValueError(f"Could not parse mask={mask!r} for statistics={statistics!r}")
    return tags_mask


def mask_for_bispec_sweep(
    statistics_row: list[str],
    tag_kb: str,
    *,
    tag_kpgm_fixed: str = "_kpgm0.25",
) -> str:
    if "bispec" in statistics_row:
        if "pgm" in statistics_row:
            return f"{tag_kb}{tag_kpgm_fixed}" if tag_kb else f"_kb{KMAX_PK_LOOSE}{tag_kpgm_fixed}"
        return tag_kb if tag_kb else f"_kb{KMAX_PK_LOOSE}"
    return tag_kpgm_fixed


def mask_for_pgm_sweep(
    statistics_row: list[str],
    tag_kpgm: str,
    *,
    tag_kb_fixed: str = "_kb0.25",
) -> str:
    if "pgm" in statistics_row:
        if "bispec" in statistics_row:
            return f"{tag_kb_fixed}{tag_kpgm}"
        return tag_kpgm
    return tag_kb_fixed


def overall_k_mask(statistics_row: list[str], k: float) -> str:
    k_other = k if k <= 0.25 else 0.25
    parts: list[str] = []
    for stat in statistics_row:
        if stat == "pk":
            parts.append(f"_kp{k}")
        elif stat == "bispec":
            parts.append(f"_kb{k_other}")
        elif stat == "pgm":
            parts.append(f"_kpgm{k_other}")
        else:
            raise ValueError(stat)
    return "".join(parts)


def iter_kmax_figure_configs() -> list[tuple[list[str], str]]:
    """Unique (statistics, joint mask) pairs used in figure_variants k_max panels."""
    configs: set[tuple[tuple[str, ...], str]] = set()
    for statistics in STAT_ROWS:
        for tag_kb in TAGS_KMAX_KB:
            configs.add((tuple(statistics), mask_for_bispec_sweep(statistics, tag_kb)))
        for tag_kpgm in TAGS_KMAX_KPGM:
            configs.add((tuple(statistics), mask_for_pgm_sweep(statistics, tag_kpgm)))
        for k in K_OVERALL:
            configs.add((tuple(statistics), overall_k_mask(statistics, k)))
    return [(list(stats), mask) for stats, mask in sorted(configs)]


def tag_inf_train(statistics: list[str], tags_mask: list[str], nth: int) -> str:
    train = _train_bundle()
    tag_data = build_tag_data(
        DATA_MODE,
        statistics,
        tags_mask,
        train["tag_params"],
        train["tag_biasparams"],
        train["tag_noise"],
    )
    base = f"{tag_data}{TAG_REPARAM}_bx{BX}_ntrain{N_TRAIN}_best{TAG_SWEEP}"
    if nth == 0:
        return base
    return f"{base}_nbest{nth}"


def model_dir(statistics: list[str], mask: str, nth: int) -> Path:
    tags_mask = parse_mask_to_tags_mask(statistics, mask)
    tag_inf = tag_inf_train(statistics, tags_mask, nth)
    return Path(paths.DIR_RESULTS) / "results_sbi" / f"sbi{tag_inf}"


def shame_test_tag(statistics: list[str], mask: str, tag_mock: str) -> str:
    tag_stats = f"_{'_'.join(statistics)}"
    return utils_plot.setup_shame_mock_test_tags(
        tag_stats_arr=[tag_stats],
        tags_mask=[mask],
        tag_mock=tag_mock,
        data_mode_test="shame",
    )[0]


def cvmean_test_tag(statistics: list[str], mask: str) -> str:
    train = _train_bundle()
    cv = resolve_test_scenario_tags("fixed_cosmo_shame_mean", NOISE_MODE, "_shame_p0_n1000")
    tag_stats = f"_{'_'.join(statistics)}"
    return utils_plot.setup_test_tags(
        data_mode=DATA_MODE,
        tag_params_test="_shame_p0_n1000",
        tags_biasparams_test=cv["tag_biasparams_test"],
        tag_stats_arr=[tag_stats],
        tag_noise_test=cv["tag_noise_test"],
        tag_datagen_test="_mean",
        tags_mask_test=[mask],
    )[0]


def member_samples_path(
    statistics: list[str],
    mask: str,
    nth: int,
    *,
    test_mode: str = "shame",
    tag_mock: str = "_nbar0.00022",
) -> Path:
    d = model_dir(statistics, mask, nth)
    if test_mode == "shame":
        tag_test = shame_test_tag(statistics, mask, tag_mock)
        return d / f"samples_test{tag_test}_pred.npy"
    if test_mode == "cvmean":
        tag_test = cvmean_test_tag(statistics, mask)
        return d / f"samples_test{tag_test}_pred.npy"
    raise ValueError(f"unknown test_mode={test_mode!r}")


def mixture_sample_equal(
    rng: np.random.Generator,
    sample_arrays: list[np.ndarray],
    n_total: int,
) -> np.ndarray:
    """Equal-weight mixture: n_total draws, each from a uniformly chosen member."""
    if not sample_arrays:
        raise ValueError("sample_arrays is empty")
    mats = []
    for arr in sample_arrays:
        a = np.asarray(arr)
        if a.ndim == 3:
            a = a[:, 0, :]
        mats.append(a)
    n_per = n_total // len(mats)
    remainder = n_total - n_per * len(mats)
    chunks = []
    for j, s in enumerate(mats):
        n_j = n_per + (1 if j < remainder else 0)
        idx = rng.choice(s.shape[0], size=n_j, replace=(n_j > s.shape[0]))
        chunks.append(s[idx])
    out = np.vstack(chunks)
    rng.shuffle(out, axis=0)
    return out


def load_ensemble_member_samples(
    statistics: list[str],
    mask: str,
    *,
    test_mode: str = "shame",
    tag_mock: str = "_nbar0.00022",
    k_members: int = N_ENSEMBLE_K,
) -> tuple[np.ndarray | None, list[str]]:
    """Load and mix top-K member sample files. Returns (samples, missing_reasons)."""
    missing: list[str] = []
    arrays: list[np.ndarray] = []
    for nth in range(k_members):
        path = member_samples_path(
            statistics, mask, nth, test_mode=test_mode, tag_mock=tag_mock,
        )
        if not path.is_file():
            missing.append(f"nbest{nth}: missing {path}")
            continue
        arr = np.load(path)
        if not np.isfinite(arr).any():
            missing.append(f"nbest{nth}: all-NaN at {path}")
            continue
        arrays.append(arr)
    if len(arrays) < k_members:
        return None, missing
    rng = np.random.default_rng(42)
    mixed = mixture_sample_equal(rng, arrays, n_total=1000)
    return mixed, []
