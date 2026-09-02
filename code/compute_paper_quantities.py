#!/usr/bin/env python3
# Run from repo root, e.g.:
#   python code/compute_paper_quantities.py
#   python code/compute_paper_quantities.py --output results/paper_quantities.dat
"""
Compute paper numbers from fiducial SBI posteriors and write a LaTeX ``.dat`` file.

Each quantity is a ``\\gdef\\MacroName{value}`` so the paper can
``\\input{.../paper_quantities.dat}`` and then use e.g.
``$\\PrecPctCVMeanPkPgmVsPkOmegaC\\%$``.

Posterior width (``err``) is the unreparameterized symmetrized inner 68%
interval of physical :math:`\\Omega_\\mathrm{c}`, :math:`\\sigma_8`,
:math:`b_1`: ``0.5*(p_{84}-p_{16})``.

Macros:
  * ``PrecPct*`` — percent precision increase, ``100*(err_ref/err_new - 1)``
  * ``RelErr*`` — relative error as a percent, ``100 * err / theta_true``
  * ``MeanErr*`` — mean posterior err over a mock set (MeanOfCVs / coverage)
  * ``FoBSigma*`` — marginal FoB on SHAMe OOD: ``|mean-truth|/sigma`` using the
    posterior covariance (same as paper-figures ``compute_fob``); ``Bo``/``Bt`` =
    :math:`b_o` (:math:`b_1`) and :math:`b_t` (:math:`b_{s2}`).

Comparisons (CV-mean and SHAMe OOD at three number densities):
  * vs :math:`P_{gg}` for :math:`P_{gg}+P_{gm}`, :math:`P_{gg}+B_{ggg}`,
    and :math:`P_{gg}+P_{gm}+B_{ggg}`
  * :math:`P_{gg}+P_{gm}+B_{ggg}` vs :math:`P_{gg}+P_{gm}`

Per-mock means of the same percent increase (currently only
:math:`P_{gg}+P_{gm}+B_{ggg}` vs :math:`P_{gg}+P_{gm}`):
  * ``MeanOfCVs`` — 1000 individual fixed-cosmo mocks
  * ``MeanOfCoverage`` — 1000 coverage-test mocks
  * ``MeanOfCoverageCenter`` — 500 coverage mocks farthest from the prior
    edges in :math:`(\\Omega_\\mathrm{c},\\sigma_8,b_1)` (closest to the
    prior-box center; same selection as the paper-figures notebook)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import data_loader  # noqa: E402
import paths  # noqa: E402
import utils_inference  # noqa: E402
import utils_model  # noqa: E402
import utils_plot  # noqa: E402

# Prior bounds used for the coverage-center cut (same as generate_params.BOUNDS /
# the paper-figures notebook extents). Avoid importing generate_params (needs bacco).
KEY_PARAM_BOUNDS = {
    "omega_cold": [0.23, 0.4],
    "sigma8_cold": [0.65, 0.9],
    "b1": [-1.0, 3.0],
}


# --- Fiducial inference (same as paper figures notebook) ---
BX = 32
N_TRAIN = 10000
TAG_INF_BEST_SUFFIX = "_best-rand30"

DATA_MODE = "muchisimocks"
TAG_PARAMS_TRAIN = "_p5_n10000"
TAG_BIASPARAMS_TRAIN = "_biasnoisenest_p9_n320000"
TAG_NOISE_TRAIN = "_noise_unit_p5_n10000"
TAG_REPARAM = "_rp"

STATISTICS_ARR_FID = [
    ["pk"],
    ["pk", "pgm"],
    ["pk", "bispec"],
    ["pk", "bispec", "pgm"],
]
TAGS_MASK_FID = ["", "_kpgm0.25", "_kb0.25", "_kb0.25_kpgm0.25"]

COMBO_KEYS = ("pk", "pk_pgm", "pk_b", "pk_pgm_b")
COMBO_LATEX = {
    "pk": "Pk",
    "pk_pgm": "PkPgm",
    "pk_b": "PkB",
    "pk_pgm_b": "PkPgmB",
}
COMBO_PLAIN = {
    "pk": "Pgg",
    "pk_pgm": "Pgg+Pgm",
    "pk_b": "Pgg+Bggg",
    "pk_pgm_b": "Pgg+Pgm+Bggg",
}

TAG_PARAMS_TEST_FIXED = "_shame_p0_n1000"
TAG_BIASPARAMS_TEST_FIXED = "_biasshame_noisebest_p0_n1"
TAG_NOISE_TEST_FIXED = "_noise_unit_shame_p0_n1000"
TAG_DATAGEN_TEST_MEAN = "_mean"

DATA_MODE_TEST_SHAME = "shame"
# SHAMe OOD mocks at three number densities (small → large n̄).
TAG_MOCKS_SHAME: Tuple[Tuple[str, str, str], ...] = (
    ("_nbar0.00011", "shame_ood_nbar11", "ShameOodNbar11"),
    ("_nbar0.00022", "shame_ood_nbar22", "ShameOodNbar22"),
    ("_nbar0.00054", "shame_ood_nbar54", "ShameOodNbar54"),
)
SHAME_NBAR_PLAIN = {
    "_nbar0.00011": "SHAMe OOD mock (n̄=1.1×10⁻⁴)",
    "_nbar0.00022": "SHAMe OOD mock (n̄=2.2×10⁻⁴)",
    "_nbar0.00054": "SHAMe OOD mock (n̄=5.4×10⁻⁴)",
}

# Marginal FoB on SHAMe for the full stats combo (Pgg+Pgm+Bggg): b_o=b1, b_t=bs2.
SHAME_FOB_COMBO = "pk_pgm_b"
SHAME_FOB_PARAMS: Tuple[Tuple[str, str, str], ...] = (
    ("b1", "Bo", "b_o (linear bias b1)"),
    ("bs2", "Bt", "b_t (tidal shear bias bs2)"),
)
FOB_RIDGE = 1e-8

TAG_PARAMS_TEST_COV = "_coverage_p5_n1000"
TAG_BIASPARAMS_TEST_COV = "_biasnoisecoverage_p9_n1000"
TAG_NOISE_TEST_COV = "_noise_unit_coverage_p5_n1000"
N_COVERAGE_CENTER = 500

IDX_OBS = 0
INF_METHOD = "sbi"

PARAM_NAMES_KEY = ("omega_cold", "sigma8_cold", "b1")
PARAM_LATEX = {
    "omega_cold": "OmegaC",
    "sigma8_cold": "SigmaEight",
    "b1": "BOne",
}
PARAM_PLAIN = {
    "omega_cold": "Omega_c",
    "sigma8_cold": "sigma_8",
    "b1": "b1",
}

DATASET_LATEX = {
    "cv_mean": "CVMean",
    "mean_of_cvs": "MeanOfCVs",
    "mean_of_coverage": "MeanOfCoverage",
    "mean_of_coverage_center": "MeanOfCoverageCenter",
    **{key: tex for _tag, key, tex in TAG_MOCKS_SHAME},
}
DATASET_PLAIN = {
    "cv_mean": "CV mean (fixed-cosmo mean of 1000 mocks)",
    "mean_of_cvs": "mean over 1000 individual CV mocks",
    "mean_of_coverage": "mean over 1000 coverage-test mocks",
    "mean_of_coverage_center": (
        f"mean over {N_COVERAGE_CENTER} coverage mocks farthest from prior edges "
        "in (Omega_c, sigma_8, b1)"
    ),
    **{key: SHAME_NBAR_PLAIN[tag] for tag, key, _tex in TAG_MOCKS_SHAME},
}

# Per-mock CV mean: only this comparison for now.
MEAN_OF_CVS_COMPARISONS = (("pk_pgm_b", "pk_pgm"),)

COMPARISONS = (
    ("pk_pgm", "pk"),
    ("pk_b", "pk"),
    ("pk_pgm_b", "pk"),
    ("pk_pgm_b", "pk_pgm"),
)


@dataclass(frozen=True)
class LatexQuantity:
    name: str
    value: float
    formatted: str
    comment: str


def _setup_fiducial_inference_tags() -> Tuple[List[str], List[str]]:
    tags_inf, _labels, _colors, tag_stats_arr = utils_plot.setup_inference_tags(
        data_mode=DATA_MODE,
        tag_params=TAG_PARAMS_TRAIN,
        tag_biasparams=TAG_BIASPARAMS_TRAIN,
        statistics_arr=STATISTICS_ARR_FID,
        bx=BX,
        tag_noise=TAG_NOISE_TRAIN,
        tag_reparam=TAG_REPARAM,
        n_train=N_TRAIN,
        tags_mask=TAGS_MASK_FID,
    )
    tags_inf = [t + TAG_INF_BEST_SUFFIX for t in tags_inf]
    return tags_inf, list(tag_stats_arr)


def _setup_cv_mean_test_tags(tag_stats_arr: Sequence[str]) -> List[str]:
    return utils_plot.setup_test_tags(
        data_mode=DATA_MODE,
        tag_params_test=TAG_PARAMS_TEST_FIXED,
        tags_biasparams_test=TAG_BIASPARAMS_TEST_FIXED,
        tag_stats_arr=tag_stats_arr,
        tag_noise_test=TAG_NOISE_TEST_FIXED,
        tag_datagen_test=TAG_DATAGEN_TEST_MEAN,
        tags_mask_test=TAGS_MASK_FID,
    )


def _setup_cv_indiv_test_tags(tag_stats_arr: Sequence[str]) -> List[str]:
    """Same fixed-cosmo test as CV mean, but the 1000 individual mocks (no ``_mean``)."""
    return utils_plot.setup_test_tags(
        data_mode=DATA_MODE,
        tag_params_test=TAG_PARAMS_TEST_FIXED,
        tags_biasparams_test=TAG_BIASPARAMS_TEST_FIXED,
        tag_stats_arr=tag_stats_arr,
        tag_noise_test=TAG_NOISE_TEST_FIXED,
        tag_datagen_test="",
        tags_mask_test=TAGS_MASK_FID,
    )


def _setup_coverage_test_tags(tag_stats_arr: Sequence[str]) -> List[str]:
    return utils_plot.setup_test_tags(
        data_mode=DATA_MODE,
        tag_params_test=TAG_PARAMS_TEST_COV,
        tags_biasparams_test=TAG_BIASPARAMS_TEST_COV,
        tag_stats_arr=tag_stats_arr,
        tag_noise_test=TAG_NOISE_TEST_COV,
        tag_datagen_test="",
        tags_mask_test=TAGS_MASK_FID,
    )


def _setup_shame_ood_test_tags(tag_mock: str) -> List[str]:
    tag_stats_arr = [
        f"_{'_'.join(stats)}{mask}"
        for stats, mask in zip(STATISTICS_ARR_FID, TAGS_MASK_FID)
    ]
    return utils_plot.setup_shame_mock_test_tags(
        tag_stats_arr=tag_stats_arr,
        data_mode_test=DATA_MODE_TEST_SHAME,
        tag_mock=tag_mock,
    )


def _sample_path(tag_inf: str, tag_test: str) -> Path:
    return (
        paths.DIR_RESULTS
        / "results_sbi"
        / f"sbi{tag_inf}"
        / f"samples_test{tag_test}_pred.npy"
    )


def _training_param_names() -> Tuple[List[str], List[str], List[str]]:
    return utils_plot.load_training_params(
        TAG_PARAMS_TRAIN, TAG_BIASPARAMS_TRAIN, bx=BX
    )


def _true_values_cv_mean(idx_obs: int = IDX_OBS) -> Dict[str, float]:
    cosmo_vary, bias_vary, param_vary = _training_param_names()
    theta = data_loader.load_theta_test(
        TAG_PARAMS_TEST_FIXED,
        TAG_BIASPARAMS_TEST_FIXED,
        cosmo_param_names_vary=cosmo_vary,
        bias_param_names_vary=bias_vary,
    )
    theta_obs = theta[idx_obs] if np.asarray(theta).ndim == 2 else np.asarray(theta)
    return {pn: float(theta_obs[param_vary.index(pn)]) for pn in PARAM_NAMES_KEY}


def _true_values_shame_ood(tag_mock: str) -> Dict[str, float]:
    # Read SHAMe truth from the stored dicts (avoids constructing a bacco cosmology).
    tag_bias = data_loader._tag_mock_shame_for_bias(tag_mock)
    theta_dict = dict(utils_model.cosmo_dict_shame)
    theta_dict.update(utils_model.bias_dict_shame[tag_bias])
    return {pn: float(theta_dict[pn]) for pn in PARAM_NAMES_KEY}


def _true_values_shame_bias(tag_mock: str, param_names: Sequence[str]) -> Dict[str, float]:
    tag_bias = data_loader._tag_mock_shame_for_bias(tag_mock)
    bias_dict = utils_model.bias_dict_shame[tag_bias]
    return {pn: float(bias_dict[pn]) for pn in param_names}


def _symmetrized_68(samples: np.ndarray, axis: int = 0) -> np.ndarray:
    """Half the 16–84 percentile range along ``axis``."""
    p16, p84 = np.percentile(samples, [16.0, 84.0], axis=axis)
    return 0.5 * (np.asarray(p84) - np.asarray(p16))


def _posterior_errs(
    tag_inf: str,
    tag_test: str,
    param_names: Sequence[str],
    idx_obs: int = IDX_OBS,
) -> Dict[str, float]:
    """Symmetrized 16–84% error after unreparameterizing to physical params."""
    path = _sample_path(tag_inf, tag_test)
    if not path.is_file():
        raise FileNotFoundError(f"Missing samples: {path}")

    samples, names = utils_inference.get_samples(
        idx_obs, INF_METHOD, tag_inf, tag_test=tag_test
    )
    if samples.size == 0 or len(names) == 0:
        raise RuntimeError(f"Empty samples for tag_inf={tag_inf!r}, tag_test={tag_test!r}")
    samples, names = utils_inference.unreparameterize_theta(samples, names)
    names_list = [str(n) for n in names]
    errs = {}
    for pn in param_names:
        if pn not in names_list:
            raise KeyError(
                f"Parameter {pn!r} not in unreparameterized chain {names_list} "
                f"(tag_inf={tag_inf!r}, tag_test={tag_test!r})"
            )
        i = names_list.index(pn)
        errs[pn] = float(_symmetrized_68(samples[:, i]))
    return errs


def _load_samples_all_obs(tag_inf: str, tag_test: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load full ``(n_posterior, n_obs, n_params)`` chain (no idx_obs slice)."""
    path = _sample_path(tag_inf, tag_test)
    if not path.is_file():
        raise FileNotFoundError(f"Missing samples: {path}")
    print(f"fn_samples (all obs) = {path}")
    samples = np.load(path)
    names = np.loadtxt(path.parent / "param_names.txt", dtype=str)
    if samples.ndim != 3:
        raise ValueError(
            f"Expected 3D samples (n_posterior, n_obs, n_params), got {samples.shape} "
            f"for {path}"
        )
    return samples, names


def _posterior_errs_per_obs(
    tag_inf: str,
    tag_test: str,
    param_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Per-observation symmetrized 16–84% err; shape ``(n_obs,)`` each."""
    samples, names = _load_samples_all_obs(tag_inf, tag_test)
    samples, names = utils_inference.unreparameterize_theta(samples, names)
    names_list = [str(n) for n in names]
    errs = {}
    for pn in param_names:
        if pn not in names_list:
            raise KeyError(
                f"Parameter {pn!r} not in unreparameterized chain {names_list} "
                f"(tag_inf={tag_inf!r}, tag_test={tag_test!r})"
            )
        i = names_list.index(pn)
        # samples: (n_posterior, n_obs, n_params)
        errs[pn] = np.asarray(_symmetrized_68(samples[:, :, i], axis=0), dtype=float)
    return errs


def _load_errs_for_combos(
    label: str,
    tags_by_combo: Mapping[str, str],
    tags_test_by_combo: Mapping[str, str],
    combos: Sequence[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    errs_by_combo: Dict[str, Dict[str, np.ndarray]] = {}
    for combo in combos:
        print(f"[{label} | {combo}] loading {tags_by_combo[combo]}  x  {tags_test_by_combo[combo]}")
        errs_by_combo[combo] = _posterior_errs_per_obs(
            tags_by_combo[combo], tags_test_by_combo[combo], PARAM_NAMES_KEY
        )
    return errs_by_combo


def _mean_prec_pct_macros(
    dataset: str,
    errs_by_combo: Mapping[str, Mapping[str, np.ndarray]],
    obs_mask: np.ndarray | None = None,
) -> List[LatexQuantity]:
    """Mean over selected observations of percent precision increase."""
    out: List[LatexQuantity] = []
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    for combo_new, combo_ref in MEAN_OF_CVS_COMPARISONS:
        for pn in PARAM_NAMES_KEY:
            err_new = np.asarray(errs_by_combo[combo_new][pn], dtype=float)
            err_ref = np.asarray(errs_by_combo[combo_ref][pn], dtype=float)
            ok = (
                np.isfinite(err_new)
                & np.isfinite(err_ref)
                & (err_new > 0)
                & (err_ref > 0)
            )
            if obs_mask is not None:
                if obs_mask.shape[0] != err_new.shape[0]:
                    raise ValueError(
                        f"obs_mask length {obs_mask.shape[0]} != n_obs {err_new.shape[0]}"
                    )
                ok = ok & np.asarray(obs_mask, dtype=bool)
            n_ok = int(np.count_nonzero(ok))
            n_tot = int(np.count_nonzero(obs_mask) if obs_mask is not None else err_new.size)
            if n_ok == 0:
                raise ValueError(
                    f"No finite per-mock errs for {dataset} {combo_new} vs {combo_ref} {pn}"
                )
            pcts = 100.0 * (err_ref[ok] / err_new[ok] - 1.0)
            pct_mean = float(np.mean(pcts))
            pct_scatter = float(np.std(pcts))
            name = (
                f"PrecPct{ds_tex}{COMBO_LATEX[combo_new]}Vs"
                f"{COMBO_LATEX[combo_ref]}{PARAM_LATEX[pn]}"
            )
            comment = (
                f"{ds_plain}: mean percent precision increase on {PARAM_PLAIN[pn]} for "
                f"{COMBO_PLAIN[combo_new]} vs {COMBO_PLAIN[combo_ref]} "
                f"(mean of 100*(err_ref/err_new-1) over {n_ok}/{n_tot} mocks; "
                f"scatter={pct_scatter:.3g})"
            )
            print(
                f"  [{dataset}] {PARAM_PLAIN[pn]}: mean {pct_mean:.3g}%  "
                f"(scatter {pct_scatter:.3g}%, n={n_ok}/{n_tot})"
            )
            out.append(LatexQuantity(name, pct_mean, _format_percent(pct_mean), comment))
    return out


def _mean_err_macros(
    dataset: str,
    errs_by_combo: Mapping[str, Mapping[str, np.ndarray]],
    obs_mask: np.ndarray | None = None,
) -> List[LatexQuantity]:
    """Mean posterior 16–84% err over selected observations, per combo/param."""
    out: List[LatexQuantity] = []
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    combos = list(errs_by_combo.keys())
    header = f"{'combo':<16}" + "".join(f"{PARAM_PLAIN[pn]:>12}" for pn in PARAM_NAMES_KEY)
    print("mean err " + header)
    for combo in combos:
        row = f"{COMBO_PLAIN[combo]:<16}"
        for pn in PARAM_NAMES_KEY:
            err = np.asarray(errs_by_combo[combo][pn], dtype=float)
            ok = np.isfinite(err) & (err > 0)
            if obs_mask is not None:
                ok = ok & np.asarray(obs_mask, dtype=bool)
            n_ok = int(np.count_nonzero(ok))
            n_tot = int(np.count_nonzero(obs_mask) if obs_mask is not None else err.size)
            if n_ok == 0:
                raise ValueError(f"No finite errs for {dataset} {combo} {pn}")
            mean_err = float(np.mean(err[ok]))
            row += f"{mean_err:12.4g}"
            name = f"MeanErr{ds_tex}{COMBO_LATEX[combo]}{PARAM_LATEX[pn]}"
            comment = (
                f"{ds_plain}: mean posterior 16-84% err on {PARAM_PLAIN[pn]} for "
                f"{COMBO_PLAIN[combo]} (mean over {n_ok}/{n_tot} mocks)"
            )
            out.append(LatexQuantity(name, mean_err, _format_err(mean_err), comment))
        print(row)
    return out


def _coverage_center_mask(n_obs: int, n_center: int = N_COVERAGE_CENTER) -> np.ndarray:
    """True for the ``n_center`` coverage mocks farthest from prior edges.

    Same construction as ``notebooks/2026-06-11_paper_figures.ipynb`` Fig 5:
    normalized Euclidean distance to the prior-box center in physical
    ``(omega_cold, sigma8_cold, b1)``.
    """
    cosmo_vary, bias_vary, param_vary = _training_param_names()
    theta = np.asarray(
        data_loader.load_theta_test(
            TAG_PARAMS_TEST_COV,
            TAG_BIASPARAMS_TEST_COV,
            cosmo_param_names_vary=cosmo_vary,
            bias_param_names_vary=bias_vary,
        ),
        dtype=float,
    )
    if theta.ndim != 2:
        raise ValueError(f"Expected 2D coverage theta, got {theta.shape}")
    if theta.shape[0] != n_obs:
        raise ValueError(
            f"Coverage theta n_obs {theta.shape[0]} != sample n_obs {n_obs}"
        )
    idxs = [list(param_vary).index(pn) for pn in PARAM_NAMES_KEY]
    theta_key = theta[:, idxs]
    bounds = [KEY_PARAM_BOUNDS[pn] for pn in PARAM_NAMES_KEY]
    center = np.array([0.5 * (lo + hi) for lo, hi in bounds], dtype=float)
    scale = np.array([0.5 * (hi - lo) for lo, hi in bounds], dtype=float)
    dist3d = np.sqrt(np.sum(((theta_key - center) / scale) ** 2, axis=1))
    idxs_center = np.argsort(dist3d)[:n_center]
    mask = np.zeros(n_obs, dtype=bool)
    mask[idxs_center] = True
    print(
        f"Coverage-center {n_center}: prior midpoints {center}, "
        f"max normalized 3D dist = {dist3d[idxs_center].max():.3f}"
    )
    return mask


def _format_percent(x: float) -> str:
    """Integer percent, no trailing junk (LaTeX text)."""
    return f"{x:.0f}"


def _format_rel_err_pct(x: float) -> str:
    """One decimal, strip trailing zeros (e.g. 7.4 or 15)."""
    s = f"{x:.1f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _format_err(x: float) -> str:
    """Compact 16–84% err width for LaTeX (4 significant figures)."""
    return f"{x:.4g}"


def _format_sigma_level(x: float) -> str:
    """One decimal FoB level (e.g. 0.6 or 1.6)."""
    s = f"{x:.1f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _posterior_marginal_fob(
    tag_inf: str,
    tag_test: str,
    param_names: Sequence[str],
    theta_true: Mapping[str, float],
    idx_obs: int = IDX_OBS,
) -> Dict[str, float]:
    """Marginal FoB = |mean - truth| / sqrt(cov_ii) after unreparameterizing."""
    samples, names = utils_inference.get_samples(
        idx_obs, INF_METHOD, tag_inf, tag_test=tag_test
    )
    if samples.size == 0 or len(names) == 0:
        raise RuntimeError(f"Empty samples for tag_inf={tag_inf!r}, tag_test={tag_test!r}")
    samples, names = utils_inference.unreparameterize_theta(samples, names)
    names_list = [str(n) for n in names]
    mu = np.mean(samples, axis=0)
    cov = np.cov(samples.T)
    fob: Dict[str, float] = {}
    for pn in param_names:
        if pn not in names_list:
            raise KeyError(
                f"Parameter {pn!r} not in unreparameterized chain {names_list} "
                f"(tag_inf={tag_inf!r}, tag_test={tag_test!r})"
            )
        if pn not in theta_true:
            raise KeyError(f"Missing truth for {pn!r}")
        i = names_list.index(pn)
        sig = float(np.sqrt(cov[i, i] + FOB_RIDGE))
        if not np.isfinite(sig) or sig <= 0:
            raise ValueError(f"Bad posterior sigma for {pn}: {sig}")
        fob[pn] = abs(float(mu[i]) - float(theta_true[pn])) / sig
    return fob


def _fob_sigma_macros(
    dataset: str,
    combo: str,
    fob_by_param: Mapping[str, float],
) -> List[LatexQuantity]:
    out: List[LatexQuantity] = []
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    combo_tex = COMBO_LATEX[combo]
    combo_plain = COMBO_PLAIN[combo]
    for pn, tex_suffix, plain_label in SHAME_FOB_PARAMS:
        val = fob_by_param[pn]
        if not np.isfinite(val):
            raise ValueError(f"Bad FoB for {dataset} {combo} {pn}: {val}")
        name = f"FoBSigma{ds_tex}{combo_tex}{tex_suffix}"
        comment = (
            f"{ds_plain}: marginal FoB (|mean-truth|/sigma) on {plain_label} for "
            f"{combo_plain} (posterior covariance diagonal)"
        )
        out.append(LatexQuantity(name, val, _format_sigma_level(val), comment))
    return out


def _prec_pct_macros(
    dataset: str,
    errs_by_combo: Mapping[str, Mapping[str, float]],
) -> List[LatexQuantity]:
    out: List[LatexQuantity] = []
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    for combo_new, combo_ref in COMPARISONS:
        for pn in PARAM_NAMES_KEY:
            err_new = errs_by_combo[combo_new][pn]
            err_ref = errs_by_combo[combo_ref][pn]
            if not np.isfinite(err_new) or not np.isfinite(err_ref) or err_new <= 0:
                raise ValueError(
                    f"Bad errs for {dataset} {combo_new} vs {combo_ref} {pn}: "
                    f"{err_new}, {err_ref}"
                )
            pct = 100.0 * (err_ref / err_new - 1.0)
            name = (
                f"PrecPct{ds_tex}{COMBO_LATEX[combo_new]}Vs"
                f"{COMBO_LATEX[combo_ref]}{PARAM_LATEX[pn]}"
            )
            comment = (
                f"{ds_plain}: percent precision increase on {PARAM_PLAIN[pn]} for "
                f"{COMBO_PLAIN[combo_new]} vs {COMBO_PLAIN[combo_ref]} "
                f"(100*(err_ref/err_new-1); err {err_ref:.4g}/{err_new:.4g})"
            )
            out.append(LatexQuantity(name, pct, _format_percent(pct), comment))
    return out


def _rel_err_macros(
    dataset: str,
    errs_by_combo: Mapping[str, Mapping[str, float]],
    theta_true: Mapping[str, float],
) -> List[LatexQuantity]:
    out: List[LatexQuantity] = []
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    for combo in COMBO_KEYS:
        for pn in PARAM_NAMES_KEY:
            true = theta_true[pn]
            if not np.isfinite(true) or true == 0:
                raise ValueError(f"Bad true value for {pn} on {dataset}: {true}")
            rel_pct = 100.0 * errs_by_combo[combo][pn] / abs(true)
            name = f"RelErr{ds_tex}{COMBO_LATEX[combo]}{PARAM_LATEX[pn]}"
            comment = (
                f"{ds_plain}: relative error (%) on {PARAM_PLAIN[pn]} for "
                f"{COMBO_PLAIN[combo]} (100*err/true; "
                f"err={errs_by_combo[combo][pn]:.4g}, true={true:.4g})"
            )
            out.append(
                LatexQuantity(name, rel_pct, _format_rel_err_pct(rel_pct), comment)
            )
    return out


def _write_dat(path: Path, quantities: Sequence[LatexQuantity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by code/compute_paper_quantities.py — do not edit by hand.",
        "% In the paper TeX file:",
        "%   \\input{<results>/paper_quantities.dat}",
        "% then e.g.",
        "%   $\\PrecPctCVMeanPkPgmVsPkOmegaC\\%$ more precise on $\\Omega_\\mathrm{c}$,",
        "%   relative error $\\RelErrCVMeanPkPgmOmegaC\\%$.",
        "%",
        "% PrecPct* = 100 * (err_ref / err_new - 1)  (percent precision increase).",
        "% RelErr*  = 100 * err / theta_true         (percent relative error).",
        "% MeanOfCVs / MeanOfCoverage / MeanOfCoverageCenter PrecPct* =",
        "%   mean of that percent increase over the corresponding mock set.",
        "% MeanErr* = mean posterior 16-84% err over that mock set.",
        "% FoBSigma* = marginal FoB |mean-truth|/sigma (posterior cov. diagonal);",
        "%   Bo/Bt = b_o (b1) and b_t (bs2) on SHAMe OOD, full stats combo only.",
        "% err = 0.5*(p84-p16) of unreparameterized posterior samples.",
        "%",
        "",
    ]
    for q in quantities:
        lines.append(f"% {q.comment}")
        lines.append(f"\\gdef\\{q.name}{{{q.formatted}}}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _print_summary(
    dataset: str,
    errs_by_combo: Mapping[str, Mapping[str, float]],
    theta_true: Mapping[str, float],
) -> None:
    print(f"\n=== {DATASET_PLAIN[dataset]} ===")
    print("true:", {pn: f"{theta_true[pn]:.4g}" for pn in PARAM_NAMES_KEY})
    header = f"{'combo':<16}" + "".join(f"{PARAM_PLAIN[pn]:>12}" for pn in PARAM_NAMES_KEY)
    print("err " + header)
    for combo in COMBO_KEYS:
        row = f"{COMBO_PLAIN[combo]:<16}"
        for pn in PARAM_NAMES_KEY:
            row += f"{errs_by_combo[combo][pn]:12.4g}"
        print(row)
    print("rel% " + header)
    for combo in COMBO_KEYS:
        row = f"{COMBO_PLAIN[combo]:<16}"
        for pn in PARAM_NAMES_KEY:
            row += f"{100.0 * errs_by_combo[combo][pn] / abs(theta_true[pn]):12.3g}"
        print(row)
    print("dPrec% " + header)
    for combo_new, combo_ref in COMPARISONS:
        label = f"{COMBO_PLAIN[combo_new]} / {COMBO_PLAIN[combo_ref]}"
        row = f"{label:<16}"
        for pn in PARAM_NAMES_KEY:
            pct = 100.0 * (
                errs_by_combo[combo_ref][pn] / errs_by_combo[combo_new][pn] - 1.0
            )
            row += f"{pct:12.3g}"
        print(row)


def compute_all(idx_obs: int = IDX_OBS) -> List[LatexQuantity]:
    tags_inf, tag_stats_arr = _setup_fiducial_inference_tags()
    tags_by_combo = dict(zip(COMBO_KEYS, tags_inf))

    tests: Dict[str, Dict[str, str]] = {
        "cv_mean": dict(zip(COMBO_KEYS, _setup_cv_mean_test_tags(tag_stats_arr))),
    }
    truths: Dict[str, Dict[str, float]] = {
        "cv_mean": _true_values_cv_mean(idx_obs=idx_obs),
    }
    for tag_mock, dataset_key, _tex in TAG_MOCKS_SHAME:
        tests[dataset_key] = dict(
            zip(COMBO_KEYS, _setup_shame_ood_test_tags(tag_mock))
        )
        truths[dataset_key] = _true_values_shame_ood(tag_mock)

    quantities: List[LatexQuantity] = []
    for dataset, tags_test in tests.items():
        errs_by_combo: Dict[str, Dict[str, float]] = {}
        for combo in COMBO_KEYS:
            tag_inf = tags_by_combo[combo]
            tag_test = tags_test[combo]
            print(f"[{dataset} | {combo}] loading {tag_inf}  x  {tag_test}")
            errs_by_combo[combo] = _posterior_errs(
                tag_inf, tag_test, PARAM_NAMES_KEY, idx_obs=idx_obs
            )
        _print_summary(dataset, errs_by_combo, truths[dataset])
        quantities.extend(_prec_pct_macros(dataset, errs_by_combo))
        quantities.extend(_rel_err_macros(dataset, errs_by_combo, truths[dataset]))

    shame_fob_params = [pn for pn, _tex, _plain in SHAME_FOB_PARAMS]
    for tag_mock, dataset_key, _tex in TAG_MOCKS_SHAME:
        tag_inf = tags_by_combo[SHAME_FOB_COMBO]
        tag_test = tests[dataset_key][SHAME_FOB_COMBO]
        truth_bias = _true_values_shame_bias(tag_mock, shame_fob_params)
        print(
            f"[{dataset_key} | {SHAME_FOB_COMBO} FoB] loading {tag_inf}  x  {tag_test}"
        )
        fob_bias = _posterior_marginal_fob(
            tag_inf, tag_test, shame_fob_params, truth_bias, idx_obs=idx_obs
        )
        for pn, _tex, plain in SHAME_FOB_PARAMS:
            print(f"  FoB {plain}: {fob_bias[pn]:.3g} sigma")
        quantities.extend(
            _fob_sigma_macros(dataset_key, SHAME_FOB_COMBO, fob_bias)
        )

    combos_needed = sorted({c for pair in MEAN_OF_CVS_COMPARISONS for c in pair})

    tags_test_indiv = dict(zip(COMBO_KEYS, _setup_cv_indiv_test_tags(tag_stats_arr)))
    print("\n=== mean over 1000 individual CV mocks ===")
    errs_cv = _load_errs_for_combos("mean_of_cvs", tags_by_combo, tags_test_indiv, combos_needed)
    quantities.extend(_mean_err_macros("mean_of_cvs", errs_cv))
    quantities.extend(_mean_prec_pct_macros("mean_of_cvs", errs_cv))

    tags_test_cov = dict(zip(COMBO_KEYS, _setup_coverage_test_tags(tag_stats_arr)))
    print("\n=== mean over coverage-test mocks ===")
    errs_cov = _load_errs_for_combos(
        "mean_of_coverage", tags_by_combo, tags_test_cov, combos_needed
    )
    quantities.extend(_mean_err_macros("mean_of_coverage", errs_cov))
    quantities.extend(_mean_prec_pct_macros("mean_of_coverage", errs_cov))
    n_obs_cov = int(np.asarray(errs_cov[combos_needed[0]][PARAM_NAMES_KEY[0]]).shape[0])
    mask_center = _coverage_center_mask(n_obs_cov, N_COVERAGE_CENTER)
    print(f"\n=== mean over {N_COVERAGE_CENTER} coverage mocks farthest from prior edges ===")
    quantities.extend(
        _mean_err_macros("mean_of_coverage_center", errs_cov, obs_mask=mask_center)
    )
    quantities.extend(
        _mean_prec_pct_macros("mean_of_coverage_center", errs_cov, obs_mask=mask_center)
    )
    return quantities


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paper precision-increase and relative-error quantities "
            "and write a LaTeX .dat file."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.DIR_RESULTS / "paper_quantities.dat",
        help="Output .dat path (default: <DIR_RESULTS>/paper_quantities.dat)",
    )
    parser.add_argument(
        "--idx-obs",
        type=int,
        default=IDX_OBS,
        help="Test-set index to use (CV-mean and SHAMe files are a single mock)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    quantities = compute_all(idx_obs=args.idx_obs)
    out = args.output
    if not out.is_absolute():
        out = REPO_ROOT / out
    _write_dat(out, quantities)
    print(f"\nWrote {len(quantities)} macros to {out}")


if __name__ == "__main__":
    main()
