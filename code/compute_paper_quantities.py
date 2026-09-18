#!/usr/bin/env python3
# Run from repo root, e.g.:
#   python code/compute_paper_quantities.py
#   python code/compute_paper_quantities.py --output figures/paper_quantities.dat
"""
Compute paper numbers from fiducial SBI posteriors and write a LaTeX ``.dat`` file.

Each quantity is a ``\\gdef\\MacroName{value}`` so the paper can
``\\input{.../paper_quantities.dat}`` and then use e.g.
``$\\PrecPctCVMeanPkPgmVsPkOmegaC\\%$``.

All posterior quantities use the equal-weight **K=3 ensemble** at the
kp0.35 fiducial masks (same as the paper figures): single-obs tests
(CV-mean, SHAMe OOD) are equal-weight mixtures; multi-obs tests
(CV-indiv, coverage) concatenate member chains along the draw axis.

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

Full-combo cosmic-variance width comparison (Appendix A):
  * ``ErrCVMean*`` — symmetrized 68% width of inference on the mean data vector
  * ``ErrCVRecentered*`` — width of the recentered pooled individual posteriors
  * ``MeanErrMeanOfCVs*`` — mean of the 1000 individual-mock widths (above)
  * ``FracDiffPctErrCVRecenteredVsCVMean*`` — ``100*(err_rc/err_mean - 1)``
  * ``FracDiffPctErrMeanOfCVsVsCVRecentered*`` — ``100*(err_indiv/err_rc - 1)``

Also writes ``\\SoftwarePackages`` — comma-separated list of external
software used in the mock-generation / inference / figure pipeline.
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
from ensemble_kmax_utils import (  # noqa: E402
    KP035_CONFIGS,
    N_ENSEMBLE_K,
    load_ensemble_member_samples,
    model_dir,
)
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

DATA_MODE = "muchisimocks"
TAG_PARAMS_TRAIN = "_p5_n10000"
TAG_BIASPARAMS_TRAIN = "_biasnoisenest_p9_n320000"

STATISTICS_ARR_FID = [list(stats) for stats, _ in KP035_CONFIGS]
TAGS_MASK_FID = [mask for _, mask in KP035_CONFIGS]

COMBO_KEYS = ("pk", "pk_pgm", "pk_b", "pk_pgm_b")
COMBO_CONFIG: Dict[str, Tuple[List[str], str]] = {
    key: (list(stats), mask)
    for key, (stats, mask) in zip(COMBO_KEYS, KP035_CONFIGS)
}
# Full stats combo used for Appendix A CV-mean vs recentered vs mean-indiv widths.
CV_WIDTH_COMBO = "pk_pgm_b"
K_PER_OBS_RECENTERED = 500
RNG_SEED_RECENTERED = 1
# Equal-weight mixture size for single-obs ensemble chains (CV-mean / SHAMe).
N_ENSEMBLE_DRAWS = 5000
# Subsample stacked multi-obs draws before unreparam (K·n_draw can be ~30k).
N_DRAW_SUBSAMPLE_MULTI = 2000
RNG_SEED_SUBSAMPLE = 0
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
    "cv_mean": f"CV mean (fixed-cosmo mean of 1000 mocks; K={N_ENSEMBLE_K} ensemble)",
    "mean_of_cvs": f"mean over 1000 individual CV mocks (K={N_ENSEMBLE_K} ensemble)",
    "mean_of_coverage": f"mean over 1000 coverage-test mocks (K={N_ENSEMBLE_K} ensemble)",
    "mean_of_coverage_center": (
        f"mean over {N_COVERAGE_CENTER} coverage mocks farthest from prior edges "
        f"in (Omega_c, sigma_8, b1) (K={N_ENSEMBLE_K} ensemble)"
    ),
    **{
        key: f"{SHAME_NBAR_PLAIN[tag]} (K={N_ENSEMBLE_K} ensemble)"
        for tag, key, _tex in TAG_MOCKS_SHAME
    },
}

# Per-mock CV mean: only this comparison for now.
MEAN_OF_CVS_COMPARISONS = (("pk_pgm_b", "pk_pgm"),)

COMPARISONS = (
    ("pk_pgm", "pk"),
    ("pk_b", "pk"),
    ("pk_pgm_b", "pk"),
    ("pk_pgm_b", "pk_pgm"),
)

# External packages used directly in this work (mock generation, statistics,
# SBI inference, hyperparameter sweeps, and paper figures). Display name for
# LaTeX, optional import name for version lookup (None = no Python package).
SOFTWARE_PACKAGES: Tuple[Tuple[str, str | None], ...] = (
    ("BACCO", "bacco"),
    ("baccoemu", "baccoemu"),
    ("map2map", None),  # external CLI (map2map_emu), not a pip import here
    ("sbi", "sbi"),
    ("PyTorch", "torch"),
    ("Weights \\& Biases", "wandb"),
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("h5py", "h5py"),
    ("pyFFTW", "pyfftw"),
    ("PyYAML", "yaml"),
    ("ChainConsumer", "chainconsumer"),
    ("emcee", "emcee"),
    ("dynesty", "dynesty"),
)


@dataclass(frozen=True)
class LatexQuantity:
    name: str
    value: object  # float for numeric macros; str for SoftwarePackages
    formatted: str
    comment: str


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


def _load_ensemble_samples(
    combo: str,
    *,
    test_mode: str,
    tag_mock: str = "_nbar0.00022",
    n_total: int = N_ENSEMBLE_DRAWS,
    allow_inprogress: bool | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Load K-member ensemble samples for one fiducial combo.

    Returns unreparameterized-ready ``(samples, param_names)`` where samples are
    2D ``(n_draw, n_params)`` for single-obs tests or 3D
    ``(n_draw, n_obs, n_params)`` for multi-obs tests.
    """
    if combo not in COMBO_CONFIG:
        raise KeyError(f"Unknown combo {combo!r}; expected one of {list(COMBO_CONFIG)}")
    statistics, mask = COMBO_CONFIG[combo]
    print(
        f"  ensemble K={N_ENSEMBLE_K} [{combo}] test_mode={test_mode}"
        + (f" tag_mock={tag_mock}" if test_mode == "shame" else "")
    )
    samples, missing = load_ensemble_member_samples(
        statistics,
        mask,
        test_mode=test_mode,
        tag_mock=tag_mock,
        k_members=N_ENSEMBLE_K,
        n_total=n_total,
        allow_inprogress=allow_inprogress,
    )
    if samples is None:
        raise FileNotFoundError(
            f"Missing ensemble samples for {combo} ({test_mode}): "
            + "; ".join(missing)
        )
    fn_pn = model_dir(statistics, mask, 0) / "param_names.txt"
    if not fn_pn.is_file():
        raise FileNotFoundError(f"Missing param_names: {fn_pn}")
    names = [str(n) for n in np.loadtxt(fn_pn, dtype=str)]
    return np.asarray(samples), names


def _errs_from_unreparam_samples(
    samples: np.ndarray,
    names: Sequence[str],
    param_names: Sequence[str],
) -> Dict[str, float]:
    """Symmetrized 16–84% err from a 2D unreparameterized chain."""
    samples_u, names_u = utils_inference.unreparameterize_theta(samples, list(names))
    names_list = [str(n) for n in names_u]
    if samples_u.ndim != 2:
        raise ValueError(f"Expected 2D samples after unreparam, got {samples_u.shape}")
    errs: Dict[str, float] = {}
    for pn in param_names:
        if pn not in names_list:
            raise KeyError(
                f"Parameter {pn!r} not in unreparameterized chain {names_list}"
            )
        i = names_list.index(pn)
        errs[pn] = float(_symmetrized_68(samples_u[:, i]))
    return errs


def _posterior_errs_ensemble(
    combo: str,
    *,
    test_mode: str,
    tag_mock: str = "_nbar0.00022",
    param_names: Sequence[str] = PARAM_NAMES_KEY,
) -> Dict[str, float]:
    """Single-obs ensemble mixture → unreparameterized symmetrized 68% errs."""
    samples, names = _load_ensemble_samples(
        combo, test_mode=test_mode, tag_mock=tag_mock, n_total=N_ENSEMBLE_DRAWS
    )
    if samples.ndim == 3:
        samples = samples[:, 0, :]
    return _errs_from_unreparam_samples(samples, names, param_names)


def _subsample_draws_3d(
    samples_3d: np.ndarray,
    n_keep: int = N_DRAW_SUBSAMPLE_MULTI,
    *,
    rng_seed: int = RNG_SEED_SUBSAMPLE,
) -> np.ndarray:
    """Subsample along the draw axis of a ``(n_draw, n_obs, n_params)`` chain."""
    samples_3d = np.asarray(samples_3d)
    if samples_3d.ndim != 3:
        raise ValueError(f"Expected 3D samples, got {samples_3d.shape}")
    n_draw = int(samples_3d.shape[0])
    k = min(int(n_keep), n_draw)
    if k >= n_draw:
        return samples_3d
    idx = np.random.default_rng(rng_seed).choice(n_draw, size=k, replace=False)
    print(f"  subsampled draws {n_draw} → {k} (seed={rng_seed})")
    return samples_3d[idx]


def _posterior_errs_per_obs_ensemble(
    combo: str,
    *,
    test_mode: str,
    tag_mock: str = "_nbar0.00022",
    param_names: Sequence[str] = PARAM_NAMES_KEY,
    samples_3d: np.ndarray | None = None,
    names: Sequence[str] | None = None,
) -> Dict[str, np.ndarray]:
    """Multi-obs ensemble (concat draws) → per-obs unreparameterized 68% errs."""
    if samples_3d is None or names is None:
        samples_3d, names = _load_ensemble_samples(
            combo, test_mode=test_mode, tag_mock=tag_mock
        )
    if samples_3d.ndim != 3:
        raise ValueError(
            f"Expected 3D multi-obs samples for {combo}/{test_mode}, got {samples_3d.shape}"
        )
    samples_3d = _subsample_draws_3d(samples_3d)
    samples_u, names_u = utils_inference.unreparameterize_theta(
        samples_3d, list(names)
    )
    names_list = [str(n) for n in names_u]
    errs: Dict[str, np.ndarray] = {}
    for pn in param_names:
        if pn not in names_list:
            raise KeyError(
                f"Parameter {pn!r} not in unreparameterized chain {names_list}"
            )
        i = names_list.index(pn)
        errs[pn] = np.asarray(_symmetrized_68(samples_u[:, :, i], axis=0), dtype=float)
    return errs


def _load_errs_for_combos_ensemble(
    label: str,
    combos: Sequence[str],
    *,
    test_mode: str,
    tag_mock: str = "_nbar0.00022",
) -> Dict[str, Dict[str, np.ndarray]]:
    errs_by_combo: Dict[str, Dict[str, np.ndarray]] = {}
    for combo in combos:
        print(f"[{label} | {combo}]")
        errs_by_combo[combo] = _posterior_errs_per_obs_ensemble(
            combo, test_mode=test_mode, tag_mock=tag_mock
        )
    return errs_by_combo


def _recentered_pooled_from_3d(
    samples_3d: np.ndarray,
    names: Sequence[str],
    param_names: Sequence[str],
    *,
    k_per_obs: int = K_PER_OBS_RECENTERED,
    rng_seed: int = RNG_SEED_RECENTERED,
    subsample: bool = True,
) -> np.ndarray:
    """Recentered pooled individual posteriors: shape ``(k * n_obs, n_params)``.

    Same construction as Appendix A: subtract each mock's posterior mean, add
    the mean of means, then stack ``k`` draws per mock.
    """
    if subsample:
        samples_3d = _subsample_draws_3d(samples_3d)
    samples_u, names_u = utils_inference.unreparameterize_theta(
        samples_3d, list(names)
    )
    names_list = [str(n) for n in names_u]
    idxs = []
    for pn in param_names:
        if pn not in names_list:
            raise KeyError(
                f"Parameter {pn!r} not in unreparameterized chain {names_list}"
            )
        idxs.append(names_list.index(pn))
    samples_sel = samples_u[:, :, idxs]
    theta_hat = np.nanmean(samples_sel, axis=0)
    finite_obs = np.all(np.isfinite(theta_hat), axis=1)
    theta_hat = theta_hat[finite_obs]
    samples_ok = samples_sel[:, finite_obs, :]
    n_obs_ok = int(theta_hat.shape[0])
    if n_obs_ok == 0:
        raise ValueError("No finite posterior means for recentering")
    theta_bar = np.mean(theta_hat, axis=0)
    n_draw = int(samples_ok.shape[0])
    k = min(int(k_per_obs), n_draw)
    draw_idx = np.random.default_rng(rng_seed).choice(n_draw, size=k, replace=False)
    s_tilde = (
        samples_ok[draw_idx]
        - theta_hat[None, :, :]
        + theta_bar[None, None, :]
    )
    pooled = s_tilde.reshape(k * n_obs_ok, len(param_names))
    print(
        f"  recentered pooled (K={N_ENSEMBLE_K}): n_obs={n_obs_ok}, k={k}, "
        f"pooled samples={pooled.shape[0]} (from {n_draw} stacked draws/obs)"
    )
    return pooled


def _errs_from_samples_2d(
    samples_2d: np.ndarray,
    param_names: Sequence[str],
) -> Dict[str, float]:
    """Symmetrized 16–84% err per column of a 2D ``(n_samples, n_params)`` chain."""
    samples_2d = np.asarray(samples_2d, dtype=float)
    if samples_2d.ndim != 2 or samples_2d.shape[1] != len(param_names):
        raise ValueError(
            f"Expected samples shape (n, {len(param_names)}), got {samples_2d.shape}"
        )
    widths = np.asarray(_symmetrized_68(samples_2d, axis=0), dtype=float)
    return {pn: float(widths[i]) for i, pn in enumerate(param_names)}


def _posterior_marginal_fob_from_samples(
    samples_2d: np.ndarray,
    names: Sequence[str],
    param_names: Sequence[str],
    theta_true: Mapping[str, float],
) -> Dict[str, float]:
    """Marginal FoB = |mean - truth| / sqrt(cov_ii) after unreparameterizing."""
    samples_u, names_u = utils_inference.unreparameterize_theta(
        samples_2d, list(names)
    )
    if samples_u.ndim != 2:
        raise ValueError(f"Expected 2D samples after unreparam, got {samples_u.shape}")
    names_list = [str(n) for n in names_u]
    mu = np.mean(samples_u, axis=0)
    cov = np.cov(samples_u.T)
    fob: Dict[str, float] = {}
    for pn in param_names:
        if pn not in names_list:
            raise KeyError(
                f"Parameter {pn!r} not in unreparameterized chain {names_list}"
            )
        if pn not in theta_true:
            raise KeyError(f"Missing truth for {pn!r}")
        i = names_list.index(pn)
        sig = float(np.sqrt(cov[i, i] + FOB_RIDGE))
        if not np.isfinite(sig) or sig <= 0:
            raise ValueError(f"Bad posterior sigma for {pn}: {sig}")
        fob[pn] = abs(float(mu[i]) - float(theta_true[pn])) / sig
    return fob


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


def _format_frac_diff_pct(x: float) -> str:
    """Signed percent fractional difference; one decimal, strip trailing zeros."""
    s = f"{x:.1f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _cv_width_comparison_macros(
    err_mean: Mapping[str, float],
    err_recentered: Mapping[str, float],
    err_mean_indiv: Mapping[str, float],
    combo: str = CV_WIDTH_COMBO,
) -> List[LatexQuantity]:
    """Widths + percent frac diffs for full-combo CV mean / recentered / mean-indiv."""
    out: List[LatexQuantity] = []
    combo_tex = COMBO_LATEX[combo]
    combo_plain = COMBO_PLAIN[combo]
    print(f"\n=== CV width comparison ({combo_plain}, K={N_ENSEMBLE_K} ensemble) ===")
    header = f"{'estimator':<22}" + "".join(
        f"{PARAM_PLAIN[pn]:>12}" for pn in PARAM_NAMES_KEY
    )
    print(header)
    rows = (
        ("inf-on-mean", err_mean),
        ("recentered pooled", err_recentered),
        ("mean of 1000 indiv", err_mean_indiv),
    )
    for label, errs in rows:
        row = f"{label:<22}"
        for pn in PARAM_NAMES_KEY:
            row += f"{errs[pn]:12.4g}"
        print(row)

    for pn in PARAM_NAMES_KEY:
        w_mean = float(err_mean[pn])
        w_rc = float(err_recentered[pn])
        w_indiv = float(err_mean_indiv[pn])
        for w, label in (
            (w_mean, "inf-on-mean"),
            (w_rc, "recentered"),
            (w_indiv, "mean-indiv"),
        ):
            if not np.isfinite(w) or w <= 0:
                raise ValueError(f"Bad {label} err for {combo} {pn}: {w}")

        frac_rc_vs_mean = 100.0 * (w_rc / w_mean - 1.0)
        frac_indiv_vs_rc = 100.0 * (w_indiv / w_rc - 1.0)
        pn_tex = PARAM_LATEX[pn]
        pn_plain = PARAM_PLAIN[pn]

        out.append(
            LatexQuantity(
                f"ErrCVMean{combo_tex}{pn_tex}",
                w_mean,
                _format_err(w_mean),
                (
                    f"CV cosmic variance, {combo_plain}: symmetrized 68% width on "
                    f"{pn_plain} from K={N_ENSEMBLE_K} ensemble inference on the "
                    f"mean data vector (err=0.5*(p84-p16))"
                ),
            )
        )
        out.append(
            LatexQuantity(
                f"ErrCVRecentered{combo_tex}{pn_tex}",
                w_rc,
                _format_err(w_rc),
                (
                    f"CV cosmic variance, {combo_plain}: symmetrized 68% width on "
                    f"{pn_plain} of K={N_ENSEMBLE_K} ensemble recentered pooled "
                    f"individual posteriors (k={K_PER_OBS_RECENTERED} draws/mock)"
                ),
            )
        )
        # MeanErrMeanOfCVs* is written separately; frac diffs use the same values.
        out.append(
            LatexQuantity(
                f"FracDiffPctErrCVRecenteredVsCVMean{combo_tex}{pn_tex}",
                frac_rc_vs_mean,
                _format_frac_diff_pct(frac_rc_vs_mean),
                (
                    f"CV cosmic variance, {combo_plain}: percent frac diff on "
                    f"{pn_plain} of recentered pooled vs inf-on-mean "
                    f"(100*(err_rc/err_mean-1); {w_rc:.4g}/{w_mean:.4g}; "
                    f"K={N_ENSEMBLE_K} ensemble)"
                ),
            )
        )
        out.append(
            LatexQuantity(
                f"FracDiffPctErrMeanOfCVsVsCVRecentered{combo_tex}{pn_tex}",
                frac_indiv_vs_rc,
                _format_frac_diff_pct(frac_indiv_vs_rc),
                (
                    f"CV cosmic variance, {combo_plain}: percent frac diff on "
                    f"{pn_plain} of mean-of-1000-indiv widths vs recentered pooled "
                    f"(100*(err_indiv/err_rc-1); {w_indiv:.4g}/{w_rc:.4g}; "
                    f"K={N_ENSEMBLE_K} ensemble)"
                ),
            )
        )
        print(
            f"  {pn_plain}: mean={w_mean:.4g}, rc={w_rc:.4g}, "
            f"indiv={w_indiv:.4g}; "
            f"frac(rc/mean-1)={frac_rc_vs_mean:.3g}%, "
            f"frac(indiv/rc-1)={frac_indiv_vs_rc:.3g}%"
        )
    return out


def _format_sigma_level(x: float) -> str:
    """One decimal FoB level (e.g. 0.6 or 1.6)."""
    s = f"{x:.1f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


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


def _package_version(import_name: str | None) -> str:
    """Best-effort installed version via metadata (no heavy package imports)."""
    if not import_name:
        return ""
    # import name → distribution name when they differ
    dist_names = {
        "yaml": "PyYAML",
        "torch": "torch",
        "PIL": "Pillow",
    }
    candidates = [dist_names.get(import_name, import_name), import_name]
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return ""
    for name in candidates:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return ""


def _software_macros() -> List[LatexQuantity]:
    """LaTeX macros listing external software packages used in this work."""
    names = [display for display, _imp in SOFTWARE_PACKAGES]
    list_tex = ", ".join(names)
    version_bits: List[str] = []
    item_lines: List[str] = []
    for display, imp in SOFTWARE_PACKAGES:
        ver = _package_version(imp)
        version_bits.append(f"{display} {ver}".rstrip() if ver else display)
        item_lines.append(
            f"\\item {display}" + (f" ({ver})" if ver else "")
        )
    versions_tex = "; ".join(version_bits)
    itemize_tex = "\n".join(item_lines)
    return [
        LatexQuantity(
            "SoftwarePackages",
            list_tex,
            list_tex,
            "External software packages used in this work (comma-separated)",
        ),
        LatexQuantity(
            "SoftwarePackagesVersions",
            versions_tex,
            versions_tex,
            "Same packages with installed versions when available (semicolon-separated)",
        ),
        LatexQuantity(
            "SoftwarePackagesItemize",
            itemize_tex,
            itemize_tex,
            "Same packages as \\item lines for a LaTeX itemize environment",
        ),
    ]


def _write_dat(path: Path, quantities: Sequence[LatexQuantity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by code/compute_paper_quantities.py — do not edit by hand.",
        "% In the paper TeX file:",
        "%   \\input{<figures>/paper_quantities.dat}",
        "% then e.g.",
        "%   $\\PrecPctCVMeanPkPgmVsPkOmegaC\\%$ more precise on $\\Omega_\\mathrm{c}$,",
        "%   relative error $\\RelErrCVMeanPkPgmOmegaC\\%$.",
        "%   Software: \\SoftwarePackages",
        "%",
        "% PrecPct* = 100 * (err_ref / err_new - 1)  (percent precision increase).",
        "% RelErr*  = 100 * err / theta_true         (percent relative error).",
        "% MeanOfCVs / MeanOfCoverage / MeanOfCoverageCenter PrecPct* =",
        "%   mean of that percent increase over the corresponding mock set.",
        "% MeanErr* = mean posterior 16-84% err over that mock set.",
        "% ErrCVMean* / ErrCVRecentered* = full-combo CV widths (inf-on-mean /",
        "%   recentered pooled indiv.); FracDiffPctErr* = 100*(err_a/err_b - 1).",
        "% FoBSigma* = marginal FoB |mean-truth|/sigma (posterior cov. diagonal);",
        "%   Bo/Bt = b_o (b1) and b_t (bs2) on SHAMe OOD, full stats combo only.",
        "% err = 0.5*(p84-p16) of unreparameterized K=3 ensemble posterior samples.",
        "% SoftwarePackages* = external packages used in mock generation,",
        "%   inference, sweeps, and paper figures.",
        "%",
        "",
    ]
    for q in quantities:
        lines.append(f"% {q.comment}")
        if q.name == "SoftwarePackagesItemize":
            # Multi-line body: keep as a single \gdef with embedded newlines.
            lines.append(f"\\gdef\\{q.name}{{%")
            for item_line in q.formatted.splitlines():
                lines.append(item_line)
            lines.append("}")
        else:
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
    truths: Dict[str, Dict[str, float]] = {
        "cv_mean": _true_values_cv_mean(idx_obs=idx_obs),
    }
    for tag_mock, dataset_key, _tex in TAG_MOCKS_SHAME:
        truths[dataset_key] = _true_values_shame_ood(tag_mock)

    quantities: List[LatexQuantity] = []
    errs_cv_mean_by_combo: Dict[str, Dict[str, float]] | None = None

    # Single-obs datasets: CV-mean + SHAMe OOD × n̄
    single_obs_jobs: List[Tuple[str, str, str]] = [
        ("cv_mean", "cvmean", "_nbar0.00022"),
    ]
    for tag_mock, dataset_key, _tex in TAG_MOCKS_SHAME:
        single_obs_jobs.append((dataset_key, "shame", tag_mock))

    for dataset, test_mode, tag_mock in single_obs_jobs:
        errs_by_combo: Dict[str, Dict[str, float]] = {}
        for combo in COMBO_KEYS:
            print(f"[{dataset} | {combo}]")
            errs_by_combo[combo] = _posterior_errs_ensemble(
                combo, test_mode=test_mode, tag_mock=tag_mock
            )
        if dataset == "cv_mean":
            errs_cv_mean_by_combo = errs_by_combo
        _print_summary(dataset, errs_by_combo, truths[dataset])
        quantities.extend(_prec_pct_macros(dataset, errs_by_combo))
        quantities.extend(_rel_err_macros(dataset, errs_by_combo, truths[dataset]))

    shame_fob_params = [pn for pn, _tex, _plain in SHAME_FOB_PARAMS]
    for tag_mock, dataset_key, _tex in TAG_MOCKS_SHAME:
        print(f"[{dataset_key} | {SHAME_FOB_COMBO} FoB]")
        samples, names = _load_ensemble_samples(
            SHAME_FOB_COMBO,
            test_mode="shame",
            tag_mock=tag_mock,
            n_total=N_ENSEMBLE_DRAWS,
        )
        if samples.ndim == 3:
            samples = samples[:, 0, :]
        truth_bias = _true_values_shame_bias(tag_mock, shame_fob_params)
        fob_bias = _posterior_marginal_fob_from_samples(
            samples, names, shame_fob_params, truth_bias
        )
        for pn, _tex, plain in SHAME_FOB_PARAMS:
            print(f"  FoB {plain}: {fob_bias[pn]:.3g} sigma")
        quantities.extend(
            _fob_sigma_macros(dataset_key, SHAME_FOB_COMBO, fob_bias)
        )

    combos_needed = sorted({c for pair in MEAN_OF_CVS_COMPARISONS for c in pair})
    if CV_WIDTH_COMBO not in combos_needed:
        combos_needed = sorted(set(combos_needed) | {CV_WIDTH_COMBO})

    print(f"\n=== mean over 1000 individual CV mocks (K={N_ENSEMBLE_K}) ===")
    errs_cv: Dict[str, Dict[str, np.ndarray]] = {}
    samples_cv_full: np.ndarray | None = None
    names_cv_full: List[str] | None = None
    for combo in combos_needed:
        print(f"[mean_of_cvs | {combo}]")
        samples_3d, names = _load_ensemble_samples(combo, test_mode="cvindiv")
        samples_sub = _subsample_draws_3d(samples_3d)
        del samples_3d
        errs_cv[combo] = _posterior_errs_per_obs_ensemble(
            combo,
            test_mode="cvindiv",
            samples_3d=samples_sub,
            names=names,
        )
        # Avoid a second unreparam subsample for the width-comparison combo.
        if combo == CV_WIDTH_COMBO:
            samples_cv_full = samples_sub
            names_cv_full = list(names)
        else:
            del samples_sub
    quantities.extend(_mean_err_macros("mean_of_cvs", errs_cv))
    quantities.extend(_mean_prec_pct_macros("mean_of_cvs", errs_cv))

    if errs_cv_mean_by_combo is None:
        raise RuntimeError("CV-mean errs were not computed")
    if samples_cv_full is None or names_cv_full is None:
        raise RuntimeError(f"Missing cvindiv samples for {CV_WIDTH_COMBO}")
    print(
        f"\n=== CV width comparison: inf-on-mean vs recentered vs mean-indiv "
        f"(K={N_ENSEMBLE_K}) ==="
    )
    pooled_rc = _recentered_pooled_from_3d(
        samples_cv_full, names_cv_full, PARAM_NAMES_KEY, subsample=False
    )
    del samples_cv_full
    err_recentered = _errs_from_samples_2d(pooled_rc, PARAM_NAMES_KEY)
    err_mean_indiv = {
        pn: float(np.mean(np.asarray(errs_cv[CV_WIDTH_COMBO][pn], dtype=float)))
        for pn in PARAM_NAMES_KEY
    }
    quantities.extend(
        _cv_width_comparison_macros(
            errs_cv_mean_by_combo[CV_WIDTH_COMBO],
            err_recentered,
            err_mean_indiv,
            combo=CV_WIDTH_COMBO,
        )
    )

    print(f"\n=== mean over coverage-test mocks (K={N_ENSEMBLE_K}) ===")
    errs_cov = _load_errs_for_combos_ensemble(
        "mean_of_coverage", combos_needed, test_mode="coverage"
    )
    quantities.extend(_mean_err_macros("mean_of_coverage", errs_cov))
    quantities.extend(_mean_prec_pct_macros("mean_of_coverage", errs_cov))
    n_obs_cov = int(np.asarray(errs_cov[combos_needed[0]][PARAM_NAMES_KEY[0]]).shape[0])
    mask_center = _coverage_center_mask(n_obs_cov, N_COVERAGE_CENTER)
    print(
        f"\n=== mean over {N_COVERAGE_CENTER} coverage mocks farthest from prior edges "
        f"(K={N_ENSEMBLE_K}) ==="
    )
    quantities.extend(
        _mean_err_macros("mean_of_coverage_center", errs_cov, obs_mask=mask_center)
    )
    quantities.extend(
        _mean_prec_pct_macros("mean_of_coverage_center", errs_cov, obs_mask=mask_center)
    )
    quantities.extend(_software_macros())
    return quantities


def _strip_software_macros(text: str) -> str:
    """Remove any previously written SoftwarePackages* macros from a .dat body."""
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("% ") and "SoftwarePackages" in line and i + 1 < len(lines):
            # Drop comment + following \\gdef\\SoftwarePackages* block
            i += 1
            if lines[i].startswith("\\gdef\\SoftwarePackages"):
                if lines[i].endswith("{%"):
                    i += 1
                    while i < len(lines) and lines[i] != "}":
                        i += 1
                    if i < len(lines) and lines[i] == "}":
                        i += 1
                else:
                    i += 1
                if i < len(lines) and lines[i] == "":
                    i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out).rstrip() + "\n"


def _append_software_to_dat(path: Path) -> int:
    """Append / refresh SoftwarePackages* macros in an existing .dat without recomputing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No existing paper quantities file at {path}; run without --software-only first."
        )
    body = _strip_software_macros(path.read_text())
    # Refresh header comments that mention SoftwarePackages if missing.
    if "SoftwarePackages*" not in body:
        body = body.replace(
            "% err = 0.5*(p84-p16) of unreparameterized posterior samples.\n%",
            "% err = 0.5*(p84-p16) of unreparameterized posterior samples.\n"
            "% SoftwarePackages* = external packages used in mock generation,\n"
            "%   inference, sweeps, and paper figures.\n%",
            1,
        )
        body = body.replace(
            "%   relative error $\\RelErrCVMeanPkPgmOmegaC\\%$.\n%",
            "%   relative error $\\RelErrCVMeanPkPgmOmegaC\\%$.\n"
            "%   Software: \\SoftwarePackages\n%",
            1,
        )
    software = _software_macros()
    extra: List[str] = [""]
    for q in software:
        extra.append(f"% {q.comment}")
        if q.name == "SoftwarePackagesItemize":
            extra.append(f"\\gdef\\{q.name}{{%")
            extra.extend(q.formatted.splitlines())
            extra.append("}")
        else:
            extra.append(f"\\gdef\\{q.name}{{{q.formatted}}}")
        extra.append("")
    path.write_text(body.rstrip() + "\n" + "\n".join(extra))
    return len(software)


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
        default=REPO_ROOT / "figures" / "paper_quantities.dat",
        help="Output .dat path (default: figures/paper_quantities.dat)",
    )
    parser.add_argument(
        "--idx-obs",
        type=int,
        default=IDX_OBS,
        help="Test-set index to use (CV-mean and SHAMe files are a single mock)",
    )
    parser.add_argument(
        "--software-only",
        action="store_true",
        help=(
            "Only (re)write SoftwarePackages* macros into an existing .dat; "
            "do not recompute posterior quantities."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out = args.output
    if not out.is_absolute():
        out = REPO_ROOT / out
    if args.software_only:
        n = _append_software_to_dat(out)
        print(f"\nWrote {n} software macros to {out}")
        return
    quantities = compute_all(idx_obs=args.idx_obs)
    _write_dat(out, quantities)
    print(f"\nWrote {len(quantities)} macros to {out}")


if __name__ == "__main__":
    main()
