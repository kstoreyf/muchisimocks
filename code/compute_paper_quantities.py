#!/usr/bin/env python3
# Run from repo root, e.g.:
#   python code/compute_paper_quantities.py
#   python code/compute_paper_quantities.py --output figures/paper_quantities.dat
# (heavy: run via sbatch, code_auxiliary/jobs/job_paper_quantities_v2.s)
"""
Compute paper numbers from fiducial SBI posteriors and write a LaTeX ``.dat`` file.

Each quantity is a ``\\gdef\\MacroName{value}`` so the paper can
``\\input{.../paper_quantities.dat}`` and then use e.g.
``$\\PrecPctCVMeanPkPgmVsPkOmegaC\\%$``.

v2 rerun (results_v2 / data_v2 trees, code_auxiliary/env_v2.sh), same setup as
``notebooks/2026-11-22_paper_figures.ipynb``: fiducial masks from
``ensemble_kmax_utils.FIDUCIAL_CONFIGS``; equal-weight **K=3 ensemble** of the
top-3 settings (``top3_settings``) for all posterior quantities, and the fresh
ensemble (``top3_fresh``) for the FoM-vs-k_max numbers (Fig. 9). Single-obs tests
(CV-mean, SHAMe OOD) are equal-weight mixtures; multi-obs tests (CV-indiv,
coverage) concatenate member chains along the draw axis.

First versions (``ALLOW_PARTIAL_ENSEMBLES``): where fewer than K members have a
test output, the leading members (top 1, or top 1+2) are used and the macro's
comment says ``[PARTIAL ensemble: ...]``. Macros whose inputs do not exist yet
are written as ``PLACEHOLDER`` (``XX``); for a precision increase both the
``PrecPct*`` and ``PrecFactor*`` names are defined then. The .dat header lists
every partial / placeholder macro.

Posterior width (``err``) is the unreparameterized symmetrized inner 68%
interval of physical :math:`\\Omega_\\mathrm{c}`, :math:`\\sigma_8`,
:math:`b_1`: ``0.5*(p_{84}-p_{16})``.

Macros:
  * ``PrecPct*`` / ``PrecFactor*`` — precision increase of one statistic combo
    vs another (percent if ≤100, else multiplicative factor ``err_ref/err_new``;
    100% → 2×, 200% → 3×); key params on all listed
    comparisons, plus bias params :math:`b_2`, :math:`b_{s2}`, :math:`b_\\ell`
    for :math:`P_{gg}+P_{gm}+B` vs :math:`P_{gg}+P_{gm}` on CV-mean and SHAMe
  * ``RelErr*`` — relative error as a percent, ``100 * err / theta_true``
  * ``MeanErr*`` — mean posterior err over a mock set (MeanOfCVs / coverage)
  * ``FoBSigma*`` — marginal FoB on SHAMe OOD: ``|mean-truth|/sigma`` using the
    posterior covariance (same as paper-figures ``compute_fob``); ``Bo``/``Bt``/``Bs`` =
    :math:`b_o` (:math:`b_1`), :math:`b_t` (:math:`b_2`), :math:`b_s` (:math:`b_{s2}`).
  * ``ShiftSigmaMatched*`` — SHAMe vs phase-matched mock (Fig. 7), full stats combo:
    signed shift of the posterior mean of :math:`\\Omega_\\mathrm{c}`, :math:`\\sigma_8`, :math:`b_1`,
    ``(mean_shame - mean_matched)/sigma_shame`` (posterior std of the SHAMe chain), at each n̄.
  * ``FoMIncrPct*`` — percent FoM increase on SHAMe OOD fid n̄ for
    :math:`\\sigma_8` pushing overall :math:`k_\\mathrm{max}` from EFT to max
    (Fig. 9; :math:`P_{gg}+P_{gm}` and full stats; macro suffix ``KmaxMaxVsKmaxEFT``).

Comparisons (CV-mean and SHAMe OOD at lo / fid / hi number density):
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
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

# v2 trees (code_auxiliary/env_v2.sh); must be set before importing paths / data_loader
os.environ["MUCHISIMOCKS_DATA"] = "/scratch/kstoreyf/muchisimocks/data_v2"
os.environ["MUCHISIMOCKS_RESULTS"] = "/scratch/kstoreyf/muchisimocks/results_v2"

import data_loader  # noqa: E402
import ensemble_kmax_utils as eku  # noqa: E402
from ensemble_kmax_utils import (  # noqa: E402
    FIDUCIAL_CONFIGS,
    N_ENSEMBLE_K,
    load_ensemble_member_samples,
    model_dir,
    TAG_PARAMS_TRAIN,
    TAG_BIASPARAMS_TRAIN,
    TAG_PARAMS_TEST_COV,
    TAG_BIASPARAMS_TEST_COV,
    TAG_PARAMS_TEST_FIXED,
    TAG_BIASPARAMS_TEST_FIXED,
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


# First versions (2026-11-24), as in notebooks/2026-11-22_paper_figures.ipynb: where fewer than K members have a
# test output, use the leading members (top 1 or top 1+2) and flag it in the macro comment. Set False for the
# final numbers (then incomplete ensembles become placeholders).
ALLOW_PARTIAL_ENSEMBLES = True
PLACEHOLDER = "XX"  # value written for macros whose inputs are not finished
# Ensemble schemes (11-22 notebook): top-3 settings for posteriors / coverage; fresh networks for FoM vs k_max.
ENSEMBLE_SCHEME_POSTERIORS = "top3_settings"
ENSEMBLE_SCHEME_KMAX = "top3_fresh"
# Fig. 9 first version uses the top network only (K=1) at every k_max point, for a consistent curve; the FoM
# increase uses the same. Set to N_ENSEMBLE_K once settings 2-3 exist at every point.
FOM_KMAX_K_MEMBERS = 1

# K members actually used per (test_mode, tag_mock, combo); filled by _load_ensemble_samples.
K_USED: Dict[Tuple[str, str, str], int] = {}

# --- Fiducial inference (same as paper figures notebook) ---
BX = 32
N_TRAIN = 10000

DATA_MODE = "muchisimocks"

STATISTICS_ARR_FID = [list(stats) for stats, _ in FIDUCIAL_CONFIGS]
TAGS_MASK_FID = [mask for _, mask in FIDUCIAL_CONFIGS]

COMBO_KEYS = ("pk", "pk_pgm", "pk_b", "pk_pgm_b")
COMBO_CONFIG: Dict[str, Tuple[List[str], str]] = {
    key: (list(stats), mask)
    for key, (stats, mask) in zip(COMBO_KEYS, FIDUCIAL_CONFIGS)
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


DATA_MODE_TEST_SHAME = "shame"
# SHAMe OOD mocks at three number densities (lo → fid → hi n̄).
TAG_MOCKS_SHAME: Tuple[Tuple[str, str, str], ...] = (
    ("_nbar0.00011", "shame_ood_lo", "ShameOodLo"),
    ("_nbar0.00022", "shame_ood_fid", "ShameOodFid"),
    ("_nbar0.00054", "shame_ood_hi", "ShameOodHi"),
)
SHAME_NBAR_PLAIN = {
    "_nbar0.00011": "SHAMe OOD mock, lo n̄ (1.1×10⁻⁴)",
    "_nbar0.00022": "SHAMe OOD mock, fid n̄ (2.2×10⁻⁴)",
    "_nbar0.00054": "SHAMe OOD mock, hi n̄ (5.4×10⁻⁴)",
}

# Marginal FoB on SHAMe for the full stats combo (Pgg+Pgm+Bggg): b_o=b1, b_t=bs2.
SHAME_FOB_COMBO = "pk_pgm_b"
SHAME_FOB_PARAMS: Tuple[Tuple[str, str, str], ...] = (
    ("b1", "Bo", "b_o (linear bias b1)"),
    ("b2", "Bt", "b_t (quadratic bias b2)"),
    ("bs2", "Bs", "b_s (tidal shear bias bs2)"),
)
FOB_RIDGE = 1e-8

# SHAMe vs phase-matched mock (Fig. 7): shift in the posterior mean of the key params, in units of the SHAMe
# posterior sigma, (mean_shame - mean_matched) / sigma_shame. Matched = SHAMe ICs + our forward model, mean of
# 10 noise realizations, same truth as SHAMe at each n̄; so the shift is the model-mismatch part of the SHAMe bias.
SHAME_MATCHED_COMBO = "pk_pgm_b"

# Fig. 9: FoM(σ8) increase when pushing overall k_max EFT→max (0.21 → 0.37; SHAMe fid n̄).
SHAME_FOM_KMAX_REF = 0.21  # EFT-scale cut (same bins as the v1 '0.2' cut)
SHAME_FOM_KMAX_HIGH = 0.37  # max Pgg scale
SHAME_FOM_KMAX_TEX = "KmaxMaxVsKmaxEFT"
SHAME_FOM_PARAM_KEY = "sigma8"  # plotter_figs FOB_KEYS name
SHAME_FOM_PARAM_PN = "sigma8_cold"
SHAME_FOM_COMBOS: Tuple[str, ...] = ("pk_pgm", "pk_pgm_b")

N_COVERAGE_CENTER = 500

IDX_OBS = 0

PARAM_NAMES_KEY = ("omega_cold", "sigma8_cold", "b1")
# Extra bias params for PkPgmB-vs-PkPgm PrecPct on CV-mean / SHAMe OOD only.
PARAM_NAMES_BIAS = ("b2", "bs2", "bl")
PARAM_NAMES_SINGLE_OBS = PARAM_NAMES_KEY + PARAM_NAMES_BIAS
PARAM_LATEX = {
    "omega_cold": "OmegaC",
    "sigma8_cold": "SigmaEight",
    "b1": "Bo",
    "b2": "Bt",
    "bs2": "BsTwo",
    "bl": "BL",
}
PARAM_PLAIN = {
    "omega_cold": "Omega_c",
    "sigma8_cold": "sigma_8",
    "b1": "b1",
    "b2": "b2",
    "bs2": "bs2",
    "bl": "bl",
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
# Bias-only PrecPct/PrecFactor: full combo vs Pgg+Pgm (CV-mean + SHAMe OOD).
BIAS_PREC_COMPARISON = ("pk_pgm_b", "pk_pgm")

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


def _mock_key(test_mode: str, tag_mock: str) -> str:
    return tag_mock if test_mode in ("shame", "matchedmean") else ""


def _k_note(test_mode: str, tag_mock: str, combos: Sequence[str]) -> str:
    """Comment suffix naming the combos that used a partial ensemble for this test."""
    parts = []
    for c in combos:
        k = K_USED.get((test_mode, _mock_key(test_mode, tag_mock), c), N_ENSEMBLE_K)
        if k < N_ENSEMBLE_K:
            parts.append(f"{COMBO_PLAIN[c]} K={k}/{N_ENSEMBLE_K}")
    return f" [PARTIAL ensemble: {', '.join(parts)}]" if parts else ""


def _placeholder(name: str, comment: str) -> LatexQuantity:
    return LatexQuantity(name, float("nan"), PLACEHOLDER, f"PLACEHOLDER (inputs not finished): {comment}")


def _prec_placeholders(name_suffix: str, comment: str) -> List[LatexQuantity]:
    """Both PrecPct* and PrecFactor* (which one the final value uses is not known yet)."""
    return [_placeholder(f"PrecPct{name_suffix}", comment),
            _placeholder(f"PrecFactor{name_suffix}", comment)]


def _try_load(label: str, fn, *args, **kwargs):
    """Run a loader; on missing inputs print why and return None (→ placeholder macros)."""
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as e:
        print(f"  MISSING [{label}]: {str(e)[:300]}")
        return None


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
    eku.PARTIAL_LOG.clear()
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
    k_used = min((k for k, _ in eku.PARTIAL_LOG.values()), default=N_ENSEMBLE_K)
    eku.PARTIAL_LOG.clear()
    K_USED[(test_mode, _mock_key(test_mode, tag_mock), combo)] = k_used
    if k_used < N_ENSEMBLE_K:
        print(f"    PARTIAL ensemble: K={k_used}/{N_ENSEMBLE_K}")
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
    errs_by_combo: Mapping[str, Mapping[str, np.ndarray] | None],
    obs_mask: np.ndarray | None = None,
    *,
    test_mode: str,
) -> List[LatexQuantity]:
    """Mean over selected observations of percent precision increase."""
    out: List[LatexQuantity] = []
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    for combo_new, combo_ref in MEAN_OF_CVS_COMPARISONS:
        note = _k_note(test_mode, "", (combo_new, combo_ref))
        for pn in PARAM_NAMES_KEY:
            if errs_by_combo.get(combo_new) is None or errs_by_combo.get(combo_ref) is None:
                out.extend(_prec_placeholders(
                    f"{ds_tex}{COMBO_LATEX[combo_new]}Vs{COMBO_LATEX[combo_ref]}{PARAM_LATEX[pn]}",
                    f"{ds_plain}: mean precision increase on {PARAM_PLAIN[pn]} for "
                    f"{COMBO_PLAIN[combo_new]} vs {COMBO_PLAIN[combo_ref]}",
                ))
                continue
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
            name_suffix = (
                f"{ds_tex}{COMBO_LATEX[combo_new]}Vs"
                f"{COMBO_LATEX[combo_ref]}{PARAM_LATEX[pn]}"
            )
            comment_pct = (
                f"{ds_plain}: mean percent precision increase on {PARAM_PLAIN[pn]} for "
                f"{COMBO_PLAIN[combo_new]} vs {COMBO_PLAIN[combo_ref]} "
                f"(mean of 100*(err_ref/err_new-1) over {n_ok}/{n_tot} mocks; "
                f"scatter={pct_scatter:.3g})" + note
            )
            comment_factor = (
                f"{ds_plain}: mean precision-increase factor on {PARAM_PLAIN[pn]} for "
                f"{COMBO_PLAIN[combo_new]} vs {COMBO_PLAIN[combo_ref]} "
                f"(mean of err_ref/err_new over {n_ok}/{n_tot} mocks; "
                f"scatter_pct={pct_scatter:.3g}; >100% → factor ×)" + note
            )
            print(
                f"  [{dataset}] {PARAM_PLAIN[pn]}: mean {pct_mean:.3g}%  "
                f"(scatter {pct_scatter:.3g}%, n={n_ok}/{n_tot})"
            )
            out.append(
                _prec_increase_quantity(
                    name_suffix=name_suffix,
                    pct=pct_mean,
                    comment_pct=comment_pct,
                    comment_factor=comment_factor,
                )
            )
    return out


def _mean_err_macros(
    dataset: str,
    errs_by_combo: Mapping[str, Mapping[str, np.ndarray] | None],
    obs_mask: np.ndarray | None = None,
    *,
    test_mode: str,
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
        if errs_by_combo[combo] is None:
            for pn in PARAM_NAMES_KEY:
                out.append(_placeholder(
                    f"MeanErr{ds_tex}{COMBO_LATEX[combo]}{PARAM_LATEX[pn]}",
                    f"{ds_plain}: mean posterior 16-84% err on {PARAM_PLAIN[pn]} for {COMBO_PLAIN[combo]}",
                ))
            print(row + "  " + PLACEHOLDER)
            continue
        note = _k_note(test_mode, "", (combo,))
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
                f"{COMBO_PLAIN[combo]} (mean over {n_ok}/{n_tot} mocks)" + note
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
    if theta.shape[0] < n_obs:
        raise ValueError(
            f"Coverage theta n_obs {theta.shape[0]} < sample n_obs {n_obs}"
        )
    idxs = [list(param_vary).index(pn) for pn in PARAM_NAMES_KEY]
    theta_key = theta[:, idxs]
    bounds = [KEY_PARAM_BOUNDS[pn] for pn in PARAM_NAMES_KEY]
    center = np.array([0.5 * (lo + hi) for lo, hi in bounds], dtype=float)
    scale = np.array([0.5 * (hi - lo) for lo, hi in bounds], dtype=float)
    dist3d = np.sqrt(np.sum(((theta_key - center) / scale) ** 2, axis=1))
    idxs_center = np.argsort(dist3d)[:n_center]
    mask = np.zeros(theta.shape[0], dtype=bool)
    mask[idxs_center] = True
    # In-progress coverage samples hold only the first n_obs rows: same center set, restricted to those.
    mask = mask[:n_obs]
    print(
        f"Coverage-center {n_center}: prior midpoints {center}, "
        f"max normalized 3D dist = {dist3d[idxs_center].max():.3f}"
    )
    return mask


def _format_percent(x: float) -> str:
    """Integer percent, no trailing junk (LaTeX text)."""
    return f"{x:.0f}"


def _format_factor(x: float) -> str:
    """Multiplicative precision factor ``err_ref/err_new``; 2 significant figures."""
    return f"{x:.2g}"


def _format_rel_err_pct(x: float) -> str:
    """One decimal, strip trailing zeros (e.g. 7.4 or 15)."""
    s = f"{x:.1f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _prec_increase_quantity(
    *,
    name_suffix: str,
    pct: float,
    comment_pct: str,
    comment_factor: str,
) -> LatexQuantity:
    """``PrecPct*`` if ≤100, else ``PrecFactor*`` = ``err_ref/err_new`` (= pct/100 + 1).

    E.g. 100% more precise → factor ``2``; 200% → factor ``3``.
    """
    if pct > 100.0:
        factor = pct / 100.0 + 1.0
        return LatexQuantity(
            f"PrecFactor{name_suffix}",
            factor,
            _format_factor(factor),
            comment_factor,
        )
    return LatexQuantity(
        f"PrecPct{name_suffix}",
        pct,
        _format_percent(pct),
        comment_pct,
    )


def _format_err(x: float) -> str:
    """Posterior 16–84% err width for LaTeX (2 significant figures)."""
    return f"{x:.2g}"


def _format_frac_diff_pct(x: float) -> str:
    """Signed percent fractional difference (1 significant figure)."""
    return f"{x:.1g}"


def _cv_width_comparison_macros(
    err_mean: Mapping[str, float] | None,
    err_recentered: Mapping[str, float] | None,
    err_mean_indiv: Mapping[str, float] | None,
    combo: str = CV_WIDTH_COMBO,
) -> List[LatexQuantity]:
    """Widths + percent frac diffs for full-combo CV mean / recentered / mean-indiv."""
    out: List[LatexQuantity] = []
    combo_tex = COMBO_LATEX[combo]
    combo_plain = COMBO_PLAIN[combo]
    if err_mean is None or err_recentered is None or err_mean_indiv is None:
        for pn in PARAM_NAMES_KEY:
            pn_tex = PARAM_LATEX[pn]
            c = f"CV cosmic variance, {combo_plain}, {PARAM_PLAIN[pn]}"
            if err_mean is None:
                out.append(_placeholder(f"ErrCVMean{combo_tex}{pn_tex}", c + ": width on mean data vector"))
            else:
                out.append(LatexQuantity(
                    f"ErrCVMean{combo_tex}{pn_tex}", float(err_mean[pn]), _format_err(float(err_mean[pn])),
                    c + ": width on mean data vector" + _k_note("cvmean", "", (combo,)),
                ))
            out.append(_placeholder(f"ErrCVRecentered{combo_tex}{pn_tex}", c + ": recentered pooled width"))
            out.append(_placeholder(f"FracDiffPctErrCVRecenteredVsCVMean{combo_tex}{pn_tex}", c))
            out.append(_placeholder(f"FracDiffPctErrMeanOfCVsVsCVRecentered{combo_tex}{pn_tex}", c))
        return out
    note = _k_note("cvmean", "", (combo,)) + _k_note("cvindiv", "", (combo,)).replace("PARTIAL", "PARTIAL cvindiv")
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
                    f"mean data vector (err=0.5*(p84-p16))" + note
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
                    f"individual posteriors (k={K_PER_OBS_RECENTERED} draws/mock)" + note
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
                    f"K={N_ENSEMBLE_K} ensemble)" + note
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
                    f"K={N_ENSEMBLE_K} ensemble)" + note
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


def _shame_vs_matched_shift_macros(
    dataset: str,
    tag_mock: str,
    combo: str = SHAME_MATCHED_COMBO,
) -> List[LatexQuantity]:
    """Signed shift of the posterior mean, SHAMe minus phase-matched, in units of sigma_SHAMe (key params)."""
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    combo_tex = COMBO_LATEX[combo]
    combo_plain = COMBO_PLAIN[combo]
    names_q = {pn: f"ShiftSigmaMatched{ds_tex}{combo_tex}{PARAM_LATEX[pn]}" for pn in PARAM_NAMES_KEY}
    desc = {pn: (f"{ds_plain}: shift in posterior mean of {PARAM_PLAIN[pn]}, SHAMe minus phase-matched mock "
                 f"(mean of 10 noise realizations), in units of the SHAMe posterior sigma, {combo_plain}")
            for pn in PARAM_NAMES_KEY}
    moments = {}
    for test_mode in ("shame", "matchedmean"):
        loaded = _try_load(f"{dataset} {test_mode} shift", _load_ensemble_samples,
                           combo, test_mode=test_mode, tag_mock=tag_mock, n_total=N_ENSEMBLE_DRAWS)
        if loaded is None:
            return [_placeholder(names_q[pn], desc[pn]) for pn in PARAM_NAMES_KEY]
        samples, names = loaded
        if samples.ndim == 3:
            samples = samples[:, 0, :]
        samples_u, names_u = utils_inference.unreparameterize_theta(samples, list(names))
        names_u = [str(n) for n in names_u]
        idx = [names_u.index(pn) for pn in PARAM_NAMES_KEY]
        moments[test_mode] = (np.mean(samples_u[:, idx], axis=0), np.std(samples_u[:, idx], axis=0, ddof=1))
    note = _k_note("shame", tag_mock, (combo,)) + _k_note("matchedmean", tag_mock, (combo,)).replace(
        "PARTIAL", "PARTIAL matched")
    out: List[LatexQuantity] = []
    for i, pn in enumerate(PARAM_NAMES_KEY):
        mu_s, sig_s = moments["shame"][0][i], moments["shame"][1][i]
        mu_m, sig_m = moments["matchedmean"][0][i], moments["matchedmean"][1][i]
        shift = float((mu_s - mu_m) / sig_s)
        print(f"  {PARAM_PLAIN[pn]}: shame {mu_s:.4g}±{sig_s:.3g}, matched {mu_m:.4g}±{sig_m:.3g} "
              f"-> shift {shift:+.2f} sigma_shame")
        out.append(LatexQuantity(
            names_q[pn], shift, _format_sigma_level(shift),
            desc[pn] + f" (mean {mu_s:.4g} vs {mu_m:.4g}; sigma_shame {sig_s:.3g}, sigma_matched {sig_m:.3g})"
            + note,
        ))
    return out


def _fob_sigma_macros(
    dataset: str,
    combo: str,
    fob_by_param: Mapping[str, float] | None,
    *,
    note: str = "",
) -> List[LatexQuantity]:
    out: List[LatexQuantity] = []
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    combo_tex = COMBO_LATEX[combo]
    combo_plain = COMBO_PLAIN[combo]
    for pn, tex_suffix, plain_label in SHAME_FOB_PARAMS:
        if fob_by_param is None:
            out.append(_placeholder(f"FoBSigma{ds_tex}{combo_tex}{tex_suffix}",
                                    f"{ds_plain}: marginal FoB on {plain_label} for {combo_plain}"))
            continue
        val = fob_by_param[pn]
        if not np.isfinite(val):
            raise ValueError(f"Bad FoB for {dataset} {combo} {pn}: {val}")
        name = f"FoBSigma{ds_tex}{combo_tex}{tex_suffix}"
        comment = (
            f"{ds_plain}: marginal FoB (|mean-truth|/sigma) on {plain_label} for "
            f"{combo_plain} (posterior covariance diagonal)" + note
        )
        out.append(LatexQuantity(name, val, _format_sigma_level(val), comment))
    return out


def _shame_fom_kmax_incr_macros() -> List[LatexQuantity]:
    """Percent FoM(σ8) increase, k_max SHAME_FOM_KMAX_HIGH vs SHAME_FOM_KMAX_REF, SHAMe fid (Fig. 9)."""
    import plotter_figs as pf  # heavy; load only when computing FoM

    tag_mock = "_nbar0.00022"
    dataset = "shame_ood_fid"
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    pn_tex = PARAM_LATEX[SHAME_FOM_PARAM_PN]
    pn_plain = PARAM_PLAIN[SHAME_FOM_PARAM_PN]
    k_ref = SHAME_FOM_KMAX_REF
    k_hi = SHAME_FOM_KMAX_HIGH
    k_mem = FOM_KMAX_K_MEMBERS
    note = (f" [{ENSEMBLE_SCHEME_KMAX}, K={k_mem}/{N_ENSEMBLE_K}: top network only, as the Fig. 9 first version]"
            if k_mem < N_ENSEMBLE_K else f" [{ENSEMBLE_SCHEME_KMAX}]")
    out: List[LatexQuantity] = []
    scheme_prev = eku.ENSEMBLE_SCHEME
    eku.set_ensemble_scheme(ENSEMBLE_SCHEME_KMAX)
    try:
        for combo in SHAME_FOM_COMBOS:
            stats, _fid_mask = COMBO_CONFIG[combo]
            mask_ref = pf.overall_k_mask(stats, k_ref)
            mask_hi = pf.overall_k_mask(stats, k_hi)
            eku.PARTIAL_LOG.clear()
            f_ref, reason_ref = pf.load_ensemble_fom_marg(
                stats, mask_ref, SHAME_FOM_PARAM_KEY, test_mode="shame", tag_mock=tag_mock, k_members=k_mem,
            )
            f_hi, reason_hi = pf.load_ensemble_fom_marg(
                stats, mask_hi, SHAME_FOM_PARAM_KEY, test_mode="shame", tag_mock=tag_mock, k_members=k_mem,
            )
            partial = sorted({k for k, _ in eku.PARTIAL_LOG.values()})
            eku.PARTIAL_LOG.clear()
            name = f"FoMIncrPct{ds_tex}{COMBO_LATEX[combo]}{pn_tex}{SHAME_FOM_KMAX_TEX}"
            desc = (f"{ds_plain}: percent FoM increase on {pn_plain} for {COMBO_PLAIN[combo]} "
                    f"at overall k_max={k_hi} (masks {mask_hi}) vs {k_ref} (masks {mask_ref})")
            if f_ref is None or f_hi is None or not np.isfinite(f_ref) or not np.isfinite(f_hi):
                print(f"  MISSING [FoM {combo}]: ref {f_ref} ({reason_ref}); hi {f_hi} ({reason_hi})")
                out.append(_placeholder(name, desc))
                continue
            pct = 100.0 * (f_hi / f_ref - 1.0)
            pnote = f" [PARTIAL ensemble: K={partial}]" if partial else ""
            comment = desc + f" (100*(FoM_hi/FoM_ref-1); FoM {f_hi:.4g}/{f_ref:.4g}; Fig. 9)" + note + pnote
            out.append(LatexQuantity(name, pct, _format_percent(pct), comment))
    finally:
        eku.set_ensemble_scheme(scheme_prev)
    return out


def _prec_pct_macros(
    dataset: str,
    errs_by_combo: Mapping[str, Mapping[str, float] | None],
    *,
    test_mode: str,
    tag_mock: str,
    param_names: Sequence[str] = PARAM_NAMES_KEY,
    comparisons: Sequence[Tuple[str, str]] = COMPARISONS,
) -> List[LatexQuantity]:
    out: List[LatexQuantity] = []
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    for combo_new, combo_ref in comparisons:
        note = _k_note(test_mode, tag_mock, (combo_new, combo_ref))
        for pn in param_names:
            if errs_by_combo.get(combo_new) is None or errs_by_combo.get(combo_ref) is None:
                out.extend(_prec_placeholders(
                    f"{ds_tex}{COMBO_LATEX[combo_new]}Vs{COMBO_LATEX[combo_ref]}{PARAM_LATEX[pn]}",
                    f"{ds_plain}: precision increase on {PARAM_PLAIN[pn]} for "
                    f"{COMBO_PLAIN[combo_new]} vs {COMBO_PLAIN[combo_ref]}",
                ))
                continue
            err_new = errs_by_combo[combo_new][pn]
            err_ref = errs_by_combo[combo_ref][pn]
            if not np.isfinite(err_new) or not np.isfinite(err_ref) or err_new <= 0:
                raise ValueError(
                    f"Bad errs for {dataset} {combo_new} vs {combo_ref} {pn}: "
                    f"{err_new}, {err_ref}"
                )
            pct = 100.0 * (err_ref / err_new - 1.0)
            name_suffix = (
                f"{ds_tex}{COMBO_LATEX[combo_new]}Vs"
                f"{COMBO_LATEX[combo_ref]}{PARAM_LATEX[pn]}"
            )
            comment_pct = (
                f"{ds_plain}: percent precision increase on {PARAM_PLAIN[pn]} for "
                f"{COMBO_PLAIN[combo_new]} vs {COMBO_PLAIN[combo_ref]} "
                f"(100*(err_ref/err_new-1); err {err_ref:.4g}/{err_new:.4g})" + note
            )
            comment_factor = (
                f"{ds_plain}: precision-increase factor on {PARAM_PLAIN[pn]} for "
                f"{COMBO_PLAIN[combo_new]} vs {COMBO_PLAIN[combo_ref]} "
                f"(err_ref/err_new; err {err_ref:.4g}/{err_new:.4g}; "
                f">100% → factor ×)" + note
            )
            out.append(
                _prec_increase_quantity(
                    name_suffix=name_suffix,
                    pct=pct,
                    comment_pct=comment_pct,
                    comment_factor=comment_factor,
                )
            )
    return out


def _rel_err_macros(
    dataset: str,
    errs_by_combo: Mapping[str, Mapping[str, float] | None],
    theta_true: Mapping[str, float],
    *,
    test_mode: str,
    tag_mock: str,
) -> List[LatexQuantity]:
    out: List[LatexQuantity] = []
    ds_tex = DATASET_LATEX[dataset]
    ds_plain = DATASET_PLAIN[dataset]
    for combo in COMBO_KEYS:
        note = _k_note(test_mode, tag_mock, (combo,))
        for pn in PARAM_NAMES_KEY:
            if errs_by_combo.get(combo) is None:
                out.append(_placeholder(f"RelErr{ds_tex}{COMBO_LATEX[combo]}{PARAM_LATEX[pn]}",
                                        f"{ds_plain}: relative error (%) on {PARAM_PLAIN[pn]} for {COMBO_PLAIN[combo]}"))
                continue
            true = theta_true[pn]
            if not np.isfinite(true) or true == 0:
                raise ValueError(f"Bad true value for {pn} on {dataset}: {true}")
            rel_pct = 100.0 * errs_by_combo[combo][pn] / abs(true)
            name = f"RelErr{ds_tex}{COMBO_LATEX[combo]}{PARAM_LATEX[pn]}"
            comment = (
                f"{ds_plain}: relative error (%) on {PARAM_PLAIN[pn]} for "
                f"{COMBO_PLAIN[combo]} (100*err/true; "
                f"err={errs_by_combo[combo][pn]:.4g}, true={true:.4g})" + note
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
    import datetime
    placeholders = [q.name for q in quantities if q.formatted == PLACEHOLDER]
    partial = [q.name for q in quantities if "PARTIAL" in q.comment or "top network only" in q.comment]
    status = [
        f"% v2 results ({os.environ['MUCHISIMOCKS_RESULTS']}), written "
        f"{datetime.datetime.now():%Y-%m-%d %H:%M}.",
        f"% Ensembles: {ENSEMBLE_SCHEME_POSTERIORS} (FoM vs k_max: {ENSEMBLE_SCHEME_KMAX}, "
        f"K={FOM_KMAX_K_MEMBERS}); partial ensembles allowed: {ALLOW_PARTIAL_ENSEMBLES}.",
        f"% STATUS: {len(placeholders)} placeholder macro(s) (value {PLACEHOLDER}, inputs not finished), "
        f"{len(partial)} macro(s) from a partial ensemble (K<{N_ENSEMBLE_K}); both will change.",
    ]
    if partial:
        status.append("%   partial: " + ", ".join(partial))
    if placeholders:
        status.append("%   placeholder: " + ", ".join(placeholders))
    lines = [
        "% Auto-generated by code/compute_paper_quantities.py — do not edit by hand.",
        *status,
        "%",
        "% In the paper TeX file:",
        "%   \\input{<figures>/paper_quantities.dat}",
        "% then e.g.",
        "%   $\\PrecPctCVMeanPkPgmVsPkOmegaC\\%$ more precise on $\\Omega_\\mathrm{c}$,",
        "%   or factor $\\PrecFactorCVMeanPkPgmVsPkSigmaEight\\times$ when >100\\%,",
        "%   relative error $\\RelErrCVMeanPkPgmOmegaC\\%$.",
        "%   Software: \\SoftwarePackages",
        "%",
        "% PrecPct*    = 100 * (err_ref / err_new - 1)  when that increase is ≤100.",
        "% PrecFactor* = err_ref / err_new               when the % increase would be >100",
        "%               (e.g. 100% → 2×, 200% → 3×).",
        "% RelErr*  = 100 * err / theta_true         (percent relative error).",
        "% MeanOfCVs / MeanOfCoverage / MeanOfCoverageCenter PrecPct*/PrecFactor* =",
        "%   mean of that increase over the corresponding mock set.",
        "% MeanErr* = mean posterior 16-84% err over that mock set.",
        "% ErrCVMean* / ErrCVRecentered* = full-combo CV widths (inf-on-mean /",
        "%   recentered pooled indiv.); FracDiffPctErr* = 100*(err_a/err_b - 1).",
        "% FoBSigma* = marginal FoB |mean-truth|/sigma (posterior cov. diagonal);",
        "%   Bo/Bt/Bs = b_o (b1), b_t (b2), b_s (bs2) on SHAMe OOD, full stats combo only.",
        "% ShiftSigmaMatched* = (mean_SHAMe - mean_matched)/sigma_SHAMe of the posterior, key params,",
        "%   SHAMe vs phase-matched mock at each n-bar (Fig. 7), full stats combo.",
        "% FoMIncrPct* = 100*(FoM_hi/FoM_ref-1) for FoM(sigma_8) on SHAMe OOD fid,",
        "%   overall k_max max vs EFT (Fig. 9; Pgg+Pgm and full stats).",
        "% err = 0.5*(p84-p16) of unreparameterized K=3 ensemble posterior samples",
        "%   (fewer members where the comment says PARTIAL).",
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
    errs_by_combo: Mapping[str, Mapping[str, float] | None],
    theta_true: Mapping[str, float],
) -> None:
    print(f"\n=== {DATASET_PLAIN[dataset]} ===")
    print("true:", {pn: f"{theta_true[pn]:.4g}" for pn in PARAM_NAMES_KEY})
    header = f"{'combo':<16}" + "".join(f"{PARAM_PLAIN[pn]:>12}" for pn in PARAM_NAMES_KEY)

    def val(combo, pn, f):
        e = errs_by_combo.get(combo)
        return f"{f(e, pn):12.4g}" if e is not None else f"{PLACEHOLDER:>12}"

    print("err " + header)
    for combo in COMBO_KEYS:
        print(f"{COMBO_PLAIN[combo]:<16}" + "".join(val(combo, pn, lambda e, p: e[p]) for pn in PARAM_NAMES_KEY))
    print("rel% " + header)
    for combo in COMBO_KEYS:
        print(f"{COMBO_PLAIN[combo]:<16}" + "".join(
            val(combo, pn, lambda e, p: 100.0 * e[p] / abs(theta_true[p])) for pn in PARAM_NAMES_KEY))
    print("dPrec% " + header)
    for combo_new, combo_ref in COMPARISONS:
        label = f"{COMBO_PLAIN[combo_new]} / {COMBO_PLAIN[combo_ref]}"
        row = f"{label:<16}"
        for pn in PARAM_NAMES_KEY:
            e_new, e_ref = errs_by_combo.get(combo_new), errs_by_combo.get(combo_ref)
            row += (f"{100.0 * (e_ref[pn] / e_new[pn] - 1.0):12.3g}" if e_new is not None and e_ref is not None
                    else f"{PLACEHOLDER:>12}")
        print(row)


def compute_all(idx_obs: int = IDX_OBS) -> List[LatexQuantity]:
    eku.set_allow_partial_ensembles(ALLOW_PARTIAL_ENSEMBLES)
    eku.set_ensemble_scheme(ENSEMBLE_SCHEME_POSTERIORS)
    K_USED.clear()
    print("results:", os.environ["MUCHISIMOCKS_RESULTS"], "| fiducial:", FIDUCIAL_CONFIGS)
    print(f"ensemble scheme {ENSEMBLE_SCHEME_POSTERIORS} (FoM vs k_max: {ENSEMBLE_SCHEME_KMAX}, "
          f"K={FOM_KMAX_K_MEMBERS}); partial ensembles allowed: {ALLOW_PARTIAL_ENSEMBLES}")
    truths: Dict[str, Dict[str, float]] = {
        "cv_mean": _true_values_cv_mean(idx_obs=idx_obs),
    }
    for tag_mock, dataset_key, _tex in TAG_MOCKS_SHAME:
        truths[dataset_key] = _true_values_shame_ood(tag_mock)

    quantities: List[LatexQuantity] = []
    errs_cv_mean_full: Dict[str, float] | None = None

    # Single-obs datasets: CV-mean + SHAMe OOD × n̄
    single_obs_jobs: List[Tuple[str, str, str]] = [
        ("cv_mean", "cvmean", "_nbar0.00022"),
    ]
    for tag_mock, dataset_key, _tex in TAG_MOCKS_SHAME:
        single_obs_jobs.append((dataset_key, "shame", tag_mock))

    for dataset, test_mode, tag_mock in single_obs_jobs:
        errs_by_combo: Dict[str, Dict[str, float] | None] = {}
        for combo in COMBO_KEYS:
            print(f"[{dataset} | {combo}]")
            errs_by_combo[combo] = _try_load(
                f"{dataset} {combo}", _posterior_errs_ensemble,
                combo, test_mode=test_mode, tag_mock=tag_mock, param_names=PARAM_NAMES_SINGLE_OBS,
            )
        if dataset == "cv_mean" and errs_by_combo[CV_WIDTH_COMBO] is not None:
            errs_cv_mean_full = {pn: errs_by_combo[CV_WIDTH_COMBO][pn] for pn in PARAM_NAMES_KEY}
        _print_summary(dataset, errs_by_combo, truths[dataset])
        quantities.extend(_prec_pct_macros(dataset, errs_by_combo, test_mode=test_mode, tag_mock=tag_mock))
        quantities.extend(
            _prec_pct_macros(
                dataset,
                errs_by_combo,
                test_mode=test_mode,
                tag_mock=tag_mock,
                param_names=PARAM_NAMES_BIAS,
                comparisons=(BIAS_PREC_COMPARISON,),
            )
        )
        quantities.extend(
            _rel_err_macros(dataset, errs_by_combo, truths[dataset], test_mode=test_mode, tag_mock=tag_mock)
        )

    shame_fob_params = [pn for pn, _tex, _plain in SHAME_FOB_PARAMS]
    for tag_mock, dataset_key, _tex in TAG_MOCKS_SHAME:
        print(f"[{dataset_key} | {SHAME_FOB_COMBO} FoB]")
        loaded = _try_load(
            f"{dataset_key} FoB", _load_ensemble_samples,
            SHAME_FOB_COMBO, test_mode="shame", tag_mock=tag_mock, n_total=N_ENSEMBLE_DRAWS,
        )
        fob_bias = None
        if loaded is not None:
            samples, names = loaded
            if samples.ndim == 3:
                samples = samples[:, 0, :]
            truth_bias = _true_values_shame_bias(tag_mock, shame_fob_params)
            fob_bias = _posterior_marginal_fob_from_samples(samples, names, shame_fob_params, truth_bias)
            for pn, _tex, plain in SHAME_FOB_PARAMS:
                print(f"  FoB {plain}: {fob_bias[pn]:.3g} sigma")
        quantities.extend(
            _fob_sigma_macros(dataset_key, SHAME_FOB_COMBO, fob_bias,
                              note=_k_note("shame", tag_mock, (SHAME_FOB_COMBO,)))
        )

    for tag_mock, dataset_key, _tex in TAG_MOCKS_SHAME:
        print(f"[{dataset_key} | {SHAME_MATCHED_COMBO} SHAMe vs phase-matched shift]")
        quantities.extend(_shame_vs_matched_shift_macros(dataset_key, tag_mock))

    print(
        f"\n=== SHAMe fid FoM(σ8) k_max {SHAME_FOM_KMAX_HIGH} vs "
        f"{SHAME_FOM_KMAX_REF} (Fig. 9) ==="
    )
    fom_incr = _shame_fom_kmax_incr_macros()
    for q in fom_incr:
        print(f"  \\{q.name}{{{q.formatted}}}")
    quantities.extend(fom_incr)

    combos_needed = sorted({c for pair in MEAN_OF_CVS_COMPARISONS for c in pair} | {CV_WIDTH_COMBO})

    print(f"\n=== mean over 1000 individual CV mocks (K={N_ENSEMBLE_K}) ===")
    errs_cv: Dict[str, Dict[str, np.ndarray] | None] = {}
    samples_cv_full: np.ndarray | None = None
    names_cv_full: List[str] | None = None
    for combo in combos_needed:
        print(f"[mean_of_cvs | {combo}]")
        loaded = _try_load(f"mean_of_cvs {combo}", _load_ensemble_samples, combo, test_mode="cvindiv")
        if loaded is None:
            errs_cv[combo] = None
            continue
        samples_3d, names = loaded
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
    quantities.extend(_mean_err_macros("mean_of_cvs", errs_cv, test_mode="cvindiv"))
    quantities.extend(_mean_prec_pct_macros("mean_of_cvs", errs_cv, test_mode="cvindiv"))

    print(
        f"\n=== CV width comparison: inf-on-mean vs recentered vs mean-indiv "
        f"(K={N_ENSEMBLE_K}) ==="
    )
    err_recentered = err_mean_indiv = None
    if samples_cv_full is not None and names_cv_full is not None:
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
            errs_cv_mean_full,
            err_recentered,
            err_mean_indiv,
            combo=CV_WIDTH_COMBO,
        )
    )

    print(f"\n=== mean over coverage-test mocks (K={N_ENSEMBLE_K}) ===")
    errs_cov: Dict[str, Dict[str, np.ndarray] | None] = {}
    for combo in combos_needed:
        print(f"[mean_of_coverage | {combo}]")
        errs_cov[combo] = _try_load(
            f"mean_of_coverage {combo}", _posterior_errs_per_obs_ensemble, combo, test_mode="coverage",
        )
    # In-progress coverage members may hold fewer rows: compare the same mocks across combos.
    n_obs_list = [np.asarray(e[PARAM_NAMES_KEY[0]]).shape[0] for e in errs_cov.values() if e is not None]
    if n_obs_list:
        n_obs_cov = int(min(n_obs_list))
        for combo, e in errs_cov.items():
            if e is not None:
                errs_cov[combo] = {pn: np.asarray(v)[:n_obs_cov] for pn, v in e.items()}
        mask_center = _coverage_center_mask(n_obs_cov, N_COVERAGE_CENTER)
    else:
        mask_center = None
    quantities.extend(_mean_err_macros("mean_of_coverage", errs_cov, test_mode="coverage"))
    quantities.extend(_mean_prec_pct_macros("mean_of_coverage", errs_cov, test_mode="coverage"))
    print(
        f"\n=== mean over {N_COVERAGE_CENTER} coverage mocks farthest from prior edges "
        f"(K={N_ENSEMBLE_K}) ==="
    )
    quantities.extend(
        _mean_err_macros("mean_of_coverage_center", errs_cov, obs_mask=mask_center, test_mode="coverage")
    )
    quantities.extend(
        _mean_prec_pct_macros("mean_of_coverage_center", errs_cov, obs_mask=mask_center, test_mode="coverage")
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
