"""Utilities for K-member ensemble inference across k_max scale cuts.

Hyperparameters come from the fiducial sweep (``TAG_SWEEP``) for each statistic combo
(``tags_mask_for_sweep``). When no sweep exists at a given k cut, the same
top-N hyperparameters are retrained at that mask (``nth_best_run`` from the
fiducial sweep).

Ensemble member 0 uses the existing ``_best{TAG_SWEEP}`` directory when present;
members 1..K-1 use ``_best{TAG_SWEEP}_nbest{n}``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import data_loader
import paths
import utils_plot
from choose_best_run import build_sweep_tag_inf
from generate_config_inference import (
    FIDUCIAL_MASKS,
    build_tag_data,
    resolve_test_scenario_tags,
    resolve_train_tag_bundle,
    TAG_PARAMS_COVERAGE,
    TAG_PARAMS_SHAME,
    TAG_SWEEP,
)

BX = 32
N_TRAIN = 10000
NOISE_MODE = "noisy"
TAG_PARAMS_TRAIN = "_p5_n10000"
DATA_MODE = "muchisimocks"
TAG_REPARAM = "_rp"

# --- Train/test tags: single source of truth for analysis + figure code (v2 sets) ---
_TRAIN_TAGS = resolve_train_tag_bundle(TAG_PARAMS_TRAIN, NOISE_MODE)
TAG_BIASPARAMS_TRAIN = _TRAIN_TAGS["tag_biasparams"]
TAG_NOISE_TRAIN = _TRAIN_TAGS["tag_noise"]

_COV_TAGS = resolve_test_scenario_tags("coverage", NOISE_MODE, TAG_PARAMS_COVERAGE)
TAG_PARAMS_TEST_COV = TAG_PARAMS_COVERAGE
TAG_BIASPARAMS_TEST_COV = _COV_TAGS["tag_biasparams_test"]
TAG_NOISE_TEST_COV = _COV_TAGS["tag_noise_test"]

_FIXED_TAGS = resolve_test_scenario_tags("fixed_cosmo_shame_mean", NOISE_MODE, TAG_PARAMS_SHAME)
TAG_PARAMS_TEST_FIXED = TAG_PARAMS_SHAME
TAG_BIASPARAMS_TEST_FIXED = _FIXED_TAGS["tag_biasparams_test"]
TAG_NOISE_TEST_FIXED = _FIXED_TAGS["tag_noise_test"]
TAG_DATAGEN_TEST_MEAN = "_mean"

# --- Scale-cut grids. Labels are data_loader.K_CUTS, shared by every statistic (a cut keeps
# bins with top edge <= label); '' = no cut (plotted at K_NO_CUT). ---
KCUT_FID_PK = data_loader.parse_kcut("pk", FIDUCIAL_MASKS["pk"])
KCUT_FID_OTHER = data_loader.parse_kcut("bispec", FIDUCIAL_MASKS["bispec"])  # B and P_gm
assert KCUT_FID_OTHER == data_loader.parse_kcut("pgm", FIDUCIAL_MASKS["pgm"])
TAGS_CUTS_KB = [data_loader.kcut_tag("bispec", k) for k in data_loader.K_CUTS]
TAGS_CUTS_KPGM = [data_loader.kcut_tag("pgm", k) for k in data_loader.K_CUTS]
TAGS_KMAX_KB = TAGS_CUTS_KB + [""]
TAGS_KMAX_KPGM = TAGS_CUTS_KPGM + [""]
K_OVERALL = list(data_loader.K_CUTS) + [data_loader.K_NO_CUT]
K_PK_FIXED_SCALE = [0.31, 0.34, 0.37]
STAT_ROWS = [["pk"], ["pk", "pgm"], ["pk", "bispec"], ["pk", "bispec", "pgm"]]
STAT_ROWS_SCALE = [["pk", "pgm"], ["pk", "bispec"], ["pk", "bispec", "pgm"]]
NBAR_TAGS_SHAME = ["_nbar0.00011", "_nbar0.00022", "_nbar0.00054"]
N_ENSEMBLE_K = 3

# Fiducial cut for each statistic combo (also where the HP sweep is run, see SWEEP_MASKS).
FIDUCIAL_CONFIGS: list[tuple[list[str], str]] = [
    (stats, "".join(FIDUCIAL_MASKS[s] for s in stats)) for stats in STAT_ROWS
]


def iter_fiducial_configs() -> list[tuple[list[str], str]]:
    """(statistics, joined mask) pairs for the fiducial-cut ensembles."""
    return [(list(stats), mask) for stats, mask in FIDUCIAL_CONFIGS]


def fiducial_mask_for_statistics(statistics: list[str]) -> str:
    """Joined fiducial mask for a statistic combo."""
    return "".join(FIDUCIAL_MASKS[s] for s in statistics)


def iter_scale_bin_configs(
    kp: float = KCUT_FID_PK,
    *,
    tags_kb: list[str] | None = None,
    tags_kpgm: list[str] | None = None,
) -> list[tuple[list[str], str, list[str]]]:
    """Scale grid at fixed ``kp``: sweep kb and kpgm over K_CUTS (no '' point).

    Returns ``(statistics, joined_mask, tags_mask)``. Deduplicates the shared
    ``pk+bispec+pgm`` point at the fiducial kb / kpgm.
    """
    tags_kb = list(tags_kb if tags_kb is not None else TAGS_CUTS_KB)
    tags_kpgm = list(tags_kpgm if tags_kpgm is not None else TAGS_CUTS_KPGM)
    seen: set[tuple[tuple[str, ...], str]] = set()
    out: list[tuple[list[str], str, list[str]]] = []

    def add(statistics: list[str], tags_mask: list[str]) -> None:
        mask = "".join(tags_mask)
        key = (tuple(statistics), mask)
        if key in seen:
            return
        seen.add(key)
        out.append((list(statistics), mask, list(tags_mask)))

    for tag_kb in tags_kb:
        add(
            ["pk", "bispec"],
            tags_mask_fixed_kp_scale_sweep(["pk", "bispec"], kp, tag_kb=tag_kb),
        )
        add(
            ["pk", "bispec", "pgm"],
            tags_mask_fixed_kp_scale_sweep(
                ["pk", "bispec", "pgm"], kp, tag_kb=tag_kb,
            ),
        )
    for tag_kpgm in tags_kpgm:
        add(
            ["pk", "pgm"],
            tags_mask_fixed_kp_scale_sweep(["pk", "pgm"], kp, tag_kpgm=tag_kpgm),
        )
        add(
            ["pk", "bispec", "pgm"],
            tags_mask_fixed_kp_scale_sweep(
                ["pk", "bispec", "pgm"], kp, tag_kpgm=tag_kpgm,
            ),
        )
    return out


def tags_mask_kp_kb_kpgm(
    kp: float,
    tag_kb: str,
    tag_kpgm: str,
) -> list[str]:
    """Per-statistic masks for joint ``pk+bispec+pgm`` at fixed ``kp`` with both cuts."""
    if not tag_kb or not tag_kpgm:
        raise ValueError("tag_kb and tag_kpgm must be non-empty for the Cartesian grid")
    return [data_loader.kcut_tag("pk", kp), tag_kb, tag_kpgm]


def iter_cartesian_bipgm_configs(
    kp: float = KCUT_FID_PK,
    *,
    tags_kb: list[str] | None = None,
    tags_kpgm: list[str] | None = None,
) -> list[tuple[list[str], str, list[str]]]:
    """Full ``kb × kpgm`` Cartesian grid for ``pk+bispec+pgm`` at fixed ``kp``.

    Default lists are all K_CUTS for both (no '' point). Returns
    ``(statistics, joined_mask, tags_mask)``.
    """
    tags_kb = list(tags_kb if tags_kb is not None else TAGS_CUTS_KB)
    tags_kpgm = list(tags_kpgm if tags_kpgm is not None else TAGS_CUTS_KPGM)
    statistics = ["pk", "bispec", "pgm"]
    out: list[tuple[list[str], str, list[str]]] = []
    for tag_kb in tags_kb:
        for tag_kpgm in tags_kpgm:
            tags_mask = tags_mask_kp_kb_kpgm(kp, tag_kb, tag_kpgm)
            out.append((list(statistics), "".join(tags_mask), tags_mask))
    return out


def sweep_name_for_tags_mask(statistics: list[str], tags_mask: list[str]) -> str:
    """Local sweep dir tag (with leading ``_``, no ``sbi`` prefix) for these masks."""
    train = _train_bundle()
    return build_sweep_tag_inf(
        data_mode=DATA_MODE,
        statistics=statistics,
        tags_mask=tags_mask,
        tag_params=train["tag_params"],
        tag_biasparams=train["tag_biasparams"],
        tag_noise=train["tag_noise"],
        reparameterize=True,
        tag_sweep=TAG_SWEEP,
    )


def _train_bundle():
    return _TRAIN_TAGS


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
    tag_kpgm_fixed: str = FIDUCIAL_MASKS["pgm"],
) -> str:
    if "bispec" in statistics_row:
        if "pgm" in statistics_row:
            return f"{tag_kb}{tag_kpgm_fixed}"
        return tag_kb
    return tag_kpgm_fixed


def mask_for_pgm_sweep(
    statistics_row: list[str],
    tag_kpgm: str,
    *,
    tag_kb_fixed: str = FIDUCIAL_MASKS["bispec"],
) -> str:
    if "pgm" in statistics_row:
        if "bispec" in statistics_row:
            return f"{tag_kb_fixed}{tag_kpgm}"
        return tag_kpgm
    return tag_kb_fixed


def overall_k_mask(statistics_row: list[str], k: float) -> str:
    """Joined mask for the overall scale cut at ``k``: every statistic is cut at ``k`` up to
    the fiducial B / P_gm cut (KCUT_FID_OTHER); above it only P(k) keeps moving."""
    k_other = min(k, KCUT_FID_OTHER)
    return "".join(
        data_loader.kcut_tag(stat, k if stat == "pk" else k_other) for stat in statistics_row
    )


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


def tags_mask_fixed_kp_scale_sweep(
    statistics: list[str],
    kp: float,
    *,
    tag_kb: str | None = None,
    tag_kpgm: str | None = None,
    tag_kb_fixed: str = FIDUCIAL_MASKS["bispec"],
    tag_kpgm_fixed: str = FIDUCIAL_MASKS["pgm"],
) -> list[str]:
    """Per-statistic masks for Fig-8-style sweeps with Pk fixed at ``kp``.

    Set ``tag_kb`` (including ``""`` for a loose bispec cut) to sweep bispec;
    set ``tag_kpgm`` to sweep PGM. Do not pass both.
    """
    if tag_kb is not None and tag_kpgm is not None:
        raise ValueError("pass tag_kb or tag_kpgm, not both")
    kp_tag = data_loader.kcut_tag("pk", kp)
    out: list[str] = []
    for stat in statistics:
        if stat == "pk":
            out.append(kp_tag)
        elif stat == "bispec":
            if tag_kb is not None:
                out.append(tag_kb)
            elif "pgm" in statistics:
                out.append(tag_kb_fixed)
            else:
                out.append("")
        elif stat == "pgm":
            if tag_kpgm is not None:
                out.append(tag_kpgm)
            elif "bispec" in statistics:
                out.append(tag_kpgm_fixed)
            else:
                out.append("")
        else:
            raise ValueError(stat)
    return out


def iter_scale_sweep_fixed_kp_configs(
    kp_values: list[float] | None = None,
) -> list[tuple[list[str], list[str]]]:
    """(statistics, tags_mask) pairs for fixed-Pk scale sweeps (Fig 8 @ kp in K_PK_FIXED_SCALE)."""
    kp_values = list(kp_values or K_PK_FIXED_SCALE)
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    configs: list[tuple[list[str], list[str]]] = []

    def add(statistics: list[str], tags_mask: list[str]) -> None:
        key = (tuple(statistics), tuple(tags_mask))
        if key in seen:
            return
        seen.add(key)
        configs.append((statistics, tags_mask))

    for kp in kp_values:
        for tag_kb in TAGS_KMAX_KB:
            add(
                ["pk", "bispec"],
                tags_mask_fixed_kp_scale_sweep(["pk", "bispec"], kp, tag_kb=tag_kb),
            )
            add(
                ["pk", "bispec", "pgm"],
                tags_mask_fixed_kp_scale_sweep(
                    ["pk", "bispec", "pgm"], kp, tag_kb=tag_kb,
                ),
            )
        for tag_kpgm in TAGS_KMAX_KPGM:
            add(
                ["pk", "pgm"],
                tags_mask_fixed_kp_scale_sweep(["pk", "pgm"], kp, tag_kpgm=tag_kpgm),
            )
            add(
                ["pk", "bispec", "pgm"],
                tags_mask_fixed_kp_scale_sweep(
                    ["pk", "bispec", "pgm"], kp, tag_kpgm=tag_kpgm,
                ),
            )

    return configs


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
    tag_stats = f"_{'_'.join(statistics)}"
    return utils_plot.setup_test_tags(
        data_mode=DATA_MODE,
        tag_params_test=TAG_PARAMS_TEST_FIXED,
        tags_biasparams_test=TAG_BIASPARAMS_TEST_FIXED,
        tag_stats_arr=[tag_stats],
        tag_noise_test=TAG_NOISE_TEST_FIXED,
        tag_datagen_test=TAG_DATAGEN_TEST_MEAN,
        tags_mask_test=[mask],
    )[0]


def cvindiv_test_tag(statistics: list[str], mask: str) -> str:
    """Fixed-cosmo shame per-realization test tag (no ``_mean``)."""
    tag_stats = f"_{'_'.join(statistics)}"
    return utils_plot.setup_test_tags(
        data_mode=DATA_MODE,
        tag_params_test=TAG_PARAMS_TEST_FIXED,
        tags_biasparams_test=TAG_BIASPARAMS_TEST_FIXED,
        tag_stats_arr=[tag_stats],
        tag_noise_test=TAG_NOISE_TEST_FIXED,
        tag_datagen_test="",
        tags_mask_test=[mask],
    )[0]


def coverage_test_tag(statistics: list[str], mask: str) -> str:
    tag_stats = f"_{'_'.join(statistics)}"
    return utils_plot.setup_test_tags(
        data_mode=DATA_MODE,
        tag_params_test=TAG_PARAMS_TEST_COV,
        tags_biasparams_test=TAG_BIASPARAMS_TEST_COV,
        tag_stats_arr=[tag_stats],
        tag_noise_test=TAG_NOISE_TEST_COV,
        tag_datagen_test="",
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
    if test_mode == "cvindiv":
        tag_test = cvindiv_test_tag(statistics, mask)
        return d / f"samples_test{tag_test}_pred.npy"
    if test_mode == "coverage":
        tag_test = coverage_test_tag(statistics, mask)
        return d / f"samples_test{tag_test}_pred.npy"
    raise ValueError(f"unknown test_mode={test_mode!r}")


def resolve_member_samples_path(
    statistics: list[str],
    mask: str,
    nth: int,
    *,
    test_mode: str = "shame",
    tag_mock: str = "_nbar0.00022",
    allow_inprogress: bool = True,
) -> tuple[Path | None, str | None]:
    """Return ``(path, kind)`` with kind in ``{'done', 'inprogress'}``, or ``(None, None)``.

    For coverage (and when ``allow_inprogress``), fall back to ``*_pred_inprogress.npy``
    if the final ``*_pred.npy`` is not written yet.
    """
    path = member_samples_path(
        statistics, mask, nth, test_mode=test_mode, tag_mock=tag_mock,
    )
    if path.is_file():
        return path, "done"
    if allow_inprogress:
        ip = path.with_name(path.name.replace("_pred.npy", "_pred_inprogress.npy"))
        if ip != path and ip.is_file():
            return ip, "inprogress"
    return None, None


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


def mixture_samples_3d_concat(sample_arrays: list[np.ndarray]) -> np.ndarray:
    """Stack K member chains along the draw axis → (K·n_draw, n_obs, n_par)."""
    if not sample_arrays:
        raise ValueError("sample_arrays is empty")
    mats = []
    for arr in sample_arrays:
        a = np.asarray(arr)
        if a.ndim == 2:
            a = a[:, np.newaxis, :]
        if a.ndim != 3:
            raise ValueError(f"expected 2D/3D samples, got shape={a.shape}")
        mats.append(a)
    n_obs = mats[0].shape[1]
    n_par = mats[0].shape[2]
    for a in mats[1:]:
        if a.shape[1] != n_obs or a.shape[2] != n_par:
            raise ValueError(
                f"member shape mismatch: {[m.shape for m in mats]}"
            )
    return np.concatenate(mats, axis=0)


def load_ensemble_member_samples(
    statistics: list[str],
    mask: str,
    *,
    test_mode: str = "shame",
    tag_mock: str = "_nbar0.00022",
    k_members: int = N_ENSEMBLE_K,
    n_total: int = 1000,
    allow_inprogress: bool | None = None,
) -> tuple[np.ndarray | None, list[str]]:
    """Load and mix top-K member sample files. Returns (samples, missing_reasons).

    For 2D / single-obs tests (shame, cvmean): equal-weight 2D mixture of size n_total.
    For multi-obs 3D tests (coverage, cvindiv): concatenate along the draw axis.

    Coverage defaults to ``allow_inprogress=True`` so partial ``*_pred_inprogress.npy``
    files can be plotted before the final ``*_pred.npy`` is written.
    """
    if allow_inprogress is None:
        allow_inprogress = test_mode == "coverage"
    missing: list[str] = []
    arrays: list[np.ndarray] = []
    kinds: list[str] = []
    for nth in range(k_members):
        path, kind = resolve_member_samples_path(
            statistics, mask, nth,
            test_mode=test_mode, tag_mock=tag_mock,
            allow_inprogress=allow_inprogress,
        )
        if path is None:
            miss = member_samples_path(
                statistics, mask, nth, test_mode=test_mode, tag_mock=tag_mock,
            )
            missing.append(f"nbest{nth}: missing {miss} (+ inprogress)" if allow_inprogress else f"nbest{nth}: missing {miss}")
            continue
        arr = np.load(path)
        if not np.isfinite(arr).any():
            missing.append(f"nbest{nth}: all-NaN at {path}")
            continue
        arrays.append(arr)
        kinds.append(kind or "done")
    if len(arrays) < k_members:
        return None, missing
    # Multi-obs 3D: keep all obs; concat draws (coverage / cvindiv).
    if any(np.asarray(a).ndim == 3 and np.asarray(a).shape[1] > 1 for a in arrays):
        # In-progress members can differ in stored n_obs; align to the shortest.
        mats = []
        for a in arrays:
            a = np.asarray(a)
            if a.ndim == 2:
                a = a[:, np.newaxis, :]
            mats.append(a)
        n_obs = min(m.shape[1] for m in mats)
        mats = [m[:, :n_obs, :] for m in mats]
        mixed = mixture_samples_3d_concat(mats)
        if any(k == "inprogress" for k in kinds):
            # Caller can detect via finite-obs count; keep arrays usable either way.
            pass
        return mixed, []
    rng = np.random.default_rng(42)
    mixed = mixture_sample_equal(rng, arrays, n_total=n_total)
    return mixed, []


def ensemble_coverage_members_exist(
    statistics: list[str], mask: str, *, k_members: int = N_ENSEMBLE_K,
    allow_inprogress: bool = True,
) -> bool:
    """True if all K members have coverage ``_pred.npy`` or (optionally) inprogress."""
    return all(
        resolve_member_samples_path(
            statistics, mask, nth, test_mode="coverage",
            allow_inprogress=allow_inprogress,
        )[0] is not None
        for nth in range(k_members)
    )
