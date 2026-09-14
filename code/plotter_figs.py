"""Paper figure plotting for mucha-sim-mocks.

Shared config at top — edit tags, masks, colors, paths here.
Notebook should: import plotter_figs as pf; pf.plot_figN_...()
"""

from __future__ import annotations

# =============================================================================
# Bootstrap (REPO_ROOT) + Imports
# =============================================================================
import pickle
import sys
from pathlib import Path

try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    REPO_ROOT = Path.cwd()
    if REPO_ROOT.name == "notebooks":
        REPO_ROOT = REPO_ROOT.parent

_CODE_DIR = str(REPO_ROOT / "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2

import paths
import plotter
import data_loader
import generate_params as genp
import utils_plot
import utils_model  # noqa: F401  (kept for notebook parity / downstream use)
import utils_inference
import utils_inference as ui

from ensemble_kmax_utils import (
    KP035_CONFIGS,
    N_ENSEMBLE_K,
    load_ensemble_member_samples,
    model_dir,
    member_samples_path,
    ensemble_coverage_members_exist,
)

# =============================================================================
# Shared configuration (edit these)
# =============================================================================
bx = 32
n_train = 10000
TAG_INF_BEST_SUFFIX = "_best-rand30"

data_mode = "muchisimocks"
TAG_PARAMS_TRAIN = "_p5_n10000"
TAG_BIASPARAMS_TRAIN = "_biasnoisenest_p9_n320000"
TAG_NOISE_TRAIN = "_noise_unit_p5_n10000"
tag_reparam = "_rp"

STATISTICS_ARR_FID = [list(stats) for stats, _ in KP035_CONFIGS]
TAGS_MASK_FID = [mask for _, mask in KP035_CONFIGS]
# pk, pk+pgm, pk+bispec, pk+bispec+pgm @ kp0.35 / kb0.25 / kpgm0.25
COLORS_FID = ["#79D0EC", "#6BB756", "#9D86E0", "#B32168"]
N_ENSEMBLE_DRAWS = 5000

STAT_LABELS_SHORT = {
    "pk": r"$P_\mathrm{gg}$",
    "bispec": r"$B_\mathrm{ggg}$",
    "pgm": r"$P_\mathrm{gm}$",
}

TAG_PARAMS_TEST_FIXED = "_shame_p0_n1000"
TAG_BIASPARAMS_TEST_FIXED = "_biasshame_noisebest_p0_n1"
TAG_NOISE_TEST_FIXED = "_noise_unit_shame_p0_n1000"
TAG_DATAGEN_TEST_MEAN = "_mean"

TAG_PARAMS_TEST_COV = "_coverage_p5_n1000"
TAG_BIASPARAMS_TEST_COV = "_biasnoisecoverage_p9_n1000"
TAG_NOISE_TEST_COV = "_noise_unit_coverage_p5_n1000"

DATA_MODE_TEST_SHAME = "shame"
TAG_MOCK_SHAME = "_nbar0.00022"

param_names_key, param_names_key_rp = utils_plot.get_param_names_key()
PARAM_NAMES_ZOOM = list(param_names_key)
idx_obs = 0

extents = {**genp.get_bounds("cosmo"), **genp.get_bounds("bias")}
extents_mid = extents.copy()
extents_mid["b1"] = [-0.4, 1.4]
extents_narrow = extents.copy()
extents_narrow["b1"] = [0.2, 0.7]
extents_contours = extents.copy()
extents_contours["b1"] = [0.0, 1.0]

FIGSIZE_CORNER = (5, 5)
FIGSIZE_ALL_PARAMS = (9, 9)
FONTSIZE_LEGEND = 12
# Contour legend (passed through plot_fiducial_contours / plot_contours_inf):
#   legend_location=(row, col)  subplot that owns the legend; (0, 1) = empty cell
#                               right of the first 1D hist on a 3-param corner
#   legend_loc='upper left'     matplotlib loc in that axes
#   loc_legend=(x, y)           bbox_to_anchor in that axes (None = no offset)
LOC_LEGEND = (1.02, 0.98)
KMAX_PK = 0.35  # fiducial P_gg cut
KMAX_LOOSE = 0.4  # empty / loose mask label on scale sweeps
ANOISE_OPTION = "Anmult"

dir_sbi = paths.DIR_RESULTS / "results_sbi"
n_cosmo_max_stats = 100  # subsample for statistics range plots

FIG_DIR = REPO_ROOT / "figures" / "2026-09-10_paper_figures"
FIG_DIR_LATEST = REPO_ROOT / "figures" / "paper_figures_latest"
SAVE_FIGURES = False  # check run: plot only, do not write PNGs
OVERWRITE_FIGURES = True
FIG_EXT = "png"

# Intermediate products for slow coverage / convergence loads (under figures/, gitignored).
CACHE_DIR = REPO_ROOT / "figures" / "paper_figures_cache"
OVERWRITE_CACHE = False  # True → recompute and overwrite existing cache files

# --- Fig 2 statistics ranges ---
COLOR_STAT = {"pk": COLORS_FID[0], "pgm": COLORS_FID[1], "bispec": COLORS_FID[2]}
SHAME_LINESTYLE = "--"
SHAME_LINEWIDTH = 1.4
LABELSIZE = 15
LEGENDSIZE = 12
TICKSIZE = 12  # axis tick labels (fig 2)
# SHAMe OOD: higher n̄ darker grey → lower n̄ lighter grey
NBAR_TAGS_STATS = ["_nbar0.00054", "_nbar0.00022", "_nbar0.00011"]
NBAR_COLORS_STATS = {
    "_nbar0.00054": "0.25",   # higher n̄ → darker
    "_nbar0.00022": "0.42",   # fiducial → mid grey
    "_nbar0.00011": "0.60",   # lower n̄ → lighter
}
# Legend entries are n̄ only; title "SHAMe mock" is set on the legend itself.
NBAR_LABELS_STATS = {
    "_nbar0.00054": r"$\bar{n}=5.4\times10^{-4}$",
    "_nbar0.00022": r"$\bar{n}=2.2\times10^{-4}$",
    "_nbar0.00011": r"$\bar{n}=1.1\times10^{-4}$",
}

# --- Fig 4 / 5 coverage ---
n_bins_fig4 = 10
PP_GAUSSIAN_SIGMA_REFS = (0.5, 0.8, 0.9, 1.1, 1.25, 2.0)
n_center = 500  # fig5 center subset size

# --- Fig 7 / 8 nbar scale ---
NBAR_TAGS_SCALE = ["_nbar0.00011", "_nbar0.00022", "_nbar0.00054"]
NBAR_LABELS = {
    "_nbar0.00011": r"$\bar{n}=1.1\times10^{-4}$",
    "_nbar0.00022": r"$\bar{n}=2.2\times10^{-4}$",
    "_nbar0.00054": r"$\bar{n}=5.4\times10^{-4}$",
}
COLOR_NOISY = COLORS_FID[3]
STATISTICS_FULL_ROW = ["pk", "bispec", "pgm"]
TAG_MASK_FULL = "_kp0.35_kb0.25_kpgm0.25"
TAG_STATS_FULL = "_pk_bispec_pgm"

# --- FoB / FoM ---
PARAM_NAMES_FOB = ["omega_cold", "sigma8_cold", "b1"]
PARAM_NAMES_FOB_S8XB1 = ["omega_cold", "sigma8_cold", "sigma8_cold_x_b1"]
FOB_KEYS = ["omega_m", "sigma8", "b1"]
FOB_KEYS_S8XB1 = ["omega_m", "sigma8", "sigma8_x_b1"]
PARAMS_TRACK = {
    "omega_m": "omega_cold",
    "sigma8": "sigma8_cold",
    "b1": "b1",
}
FOB_LABELS = {
    "omega_m": r"$\Omega_\mathrm{c}$",
    "sigma8": r"$\sigma_8$",
    "b1": r"$b_1$",
    "sigma8_x_b1": r"$\sigma_8 b_1$",
}
_CHI2_PPF_1SIG = 0.682689492137
FOB_RIDGE = 1e-8
FOB3_YLABEL = r"FoB($\Omega_m$, $\sigma_8$, $b_1$)"
FOB3_YLABEL_S8XB1 = r"FoB($\Omega_m$, $\sigma_8$, $\sigma_8 b_1$)"
FOM3_YLABEL = r"FoM($\Omega_m$, $\sigma_8$, $b_1$)"
FOM_FULL_YLABEL = r"FoM(all params)"
FOM_MARG_LABELS = {
    "omega_m": r"FoM($\Omega_\mathrm{c}$)",
    "sigma8": r"FoM($\sigma_8$)",
    "b1": r"FoM($b_1$)",
}

# --- Fig 8 scale dependence (fixed kp=0.35) ---
TAGS_KMAX_KPGM = ["_kpgm0.1", "_kpgm0.15", "_kpgm0.2", "_kpgm0.25", "_kpgm0.27", "_kpgm0.32", "_kpgm0.37"]
TAGS_KMAX_KB = ["_kb0.1", "_kb0.15", "_kb0.2", "_kb0.25", "_kb0.27", "_kb0.32", "_kb0.37"]
STAT_COMBOS_SCALE = [
    (["pk", "pgm"], COLORS_FID[1]),
    (["pk", "bispec"], COLORS_FID[2]),
    (["pk", "bispec", "pgm"], COLORS_FID[3]),
]

# --- Fig 9 overall k_max ---
_K_MIN_STAT = 0.01
_K_MAX_STAT = 0.4
_N_BINS_PK = 32
K_EDGES_PK = np.logspace(np.log10(_K_MIN_STAT), np.log10(_K_MAX_STAT), _N_BINS_PK + 1)
K_CENTERS_PK = np.sqrt(K_EDGES_PK[:-1] * K_EDGES_PK[1:])
K_OVERALL = [0.1, 0.15, 0.2, 0.25, 0.27, 0.32, 0.37]
STAT_COMBOS_OVERALL = [
    (list(stats), COLORS_FID[i]) for i, (stats, _) in enumerate(KP035_CONFIGS)
]
KMAX_OVERALL_XLABEL = "\n".join([
    r"$k_{\mathrm{max}}^{\mathrm{all}}\ (\leq 0.25)$",
    r"$k_{\mathrm{max}}^{P_{\mathrm{gg}}}\ (>0.25;\ "
    r"k_{\mathrm{max}}^{B}=k_{\mathrm{max}}^{P_{\mathrm{gm}}}=0.25)$",
])
KMAX_XLIM = (0.09, 0.375)  # shared by fig 8 / 9

# --- Fig 10 / 11 convergence ---
n_cosmo_arr = [500, 1000, 2000, 4000, 6000, 8000, 10000]
bx_arr = [1, 2, 4, 8, 16, 32]
N_COVERAGE = 1000
NCOSMO_ALPHA_LO, NCOSMO_ALPHA_HI = 0.15, 1.0
XLABEL_NSIMS = r"$N_\mathrm{train}$ ($= N_\mathrm{cosmo} \times N_\mathrm{bias}$)"
STATISTICS_FULL_CONV = ["pk", "bispec", "pgm"]
TAG_MASKS_FULL_CONV = "_kp0.35_kb0.25_kpgm0.25"

# --- Appendix A (9-8 A1 style) ---
COLOR_INDIV_MEANS = "black"       # optional: distribution of posterior means
COLOR_RECENTERED = "#8B5A2B"      # brown: recentered pooled + mean-of-means mark
COLOR_RAND = "#8C8C8C"            # grey: random individual posteriors
LW_MEAN_A = 1.2
LW_RECENTERED_A = 1.6
LW_RAND_A = 0.35
LW_CONTOUR_A = LW_MEAN_A  # alias
N_RAND_A = 10
SMOOTH_MEANS = 0
BINS_MEANS = 12
KDE_MEANS = True
K_PER_OBS = 500
EXTENTS_A = {
    "omega_cold": [0.25, 0.37],
    "sigma8_cold": [0.75, 0.88],
    "b1": [0.35, 0.60],
}
LABEL_MEAN_A = "inference on mean data vector"
LABEL_INDIV_A = "distribution of means of indiv. posteriors"
LABEL_MEAN_OF_MEANS_A = "mean of individual posteriors"
LABEL_RECENTERED_A = "recentered, pooled indiv. posteriors"
LABEL_RAND_A = rf"{N_RAND_A} random indiv. posteriors"


# =============================================================================
# Internal helpers (setup, cache, save, FoB/FoM, ensemble load, etc.)
# =============================================================================

def get_stat_label_short(statistics):
    # Display order: Pgg, Pgm, Bggg (independent of tag / file-name order).
    order = {"pk": 0, "pgm": 1, "bispec": 2}
    stats = sorted(statistics, key=lambda s: order.get(s, 99))
    return " + ".join(STAT_LABELS_SHORT[s] for s in stats)


def load_or_build(name, builder, *, overwrite=None):
    """Load pickle from CACHE_DIR if present; else run builder(), save, and return.

    Set OVERWRITE_CACHE=True (or pass overwrite=True) to force a rebuild.
    """
    if overwrite is None:
        overwrite = OVERWRITE_CACHE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.pkl"
    if path.is_file() and not overwrite:
        print(f"Loaded cache: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = builder()
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote cache: {path}")
    return obj


def save_figure(fig, name, ext=FIG_EXT):
    """Save figure to FIG_DIR and FIG_DIR_LATEST; skip if exists and OVERWRITE_FIGURES is False."""
    if not SAVE_FIGURES:
        return
    filename = f"{name}.{ext}"
    paths_out = [FIG_DIR / filename, FIG_DIR_LATEST / filename]
    if any(p.exists() for p in paths_out) and not OVERWRITE_FIGURES:
        print(f"Skipping save (exists, overwrite=False): {filename}")
        return
    # Ensure artists are drawn; opaque white avoids blank/black PNGs from transparent bg.
    fig.canvas.draw_idle()
    for path in paths_out:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Saved: {path}")


def save_current_figure(name, ext=FIG_EXT):
    """Deprecated: prefer save_figure(fig, name) before plt.show()."""
    save_figure(plt.gcf(), name, ext=ext)


def setup_fiducial_inference():
    tags_inf, labels, colors, _ = utils_plot.setup_inference_tags(
        data_mode=data_mode,
        tag_params=TAG_PARAMS_TRAIN,
        tag_biasparams=TAG_BIASPARAMS_TRAIN,
        statistics_arr=STATISTICS_ARR_FID,
        bx=bx,
        tag_noise=TAG_NOISE_TRAIN,
        tag_reparam=tag_reparam,
        n_train=n_train,
        tags_mask=TAGS_MASK_FID,
    )
    tags_inf = [t + TAG_INF_BEST_SUFFIX for t in tags_inf]
    labels = [get_stat_label_short(s) for s in STATISTICS_ARR_FID]
    return tags_inf, labels, list(COLORS_FID)


def sample_fn(tag_inf, tag_test):
    return dir_sbi / f"sbi{tag_inf}" / f"samples_test{tag_test}_pred.npy"


def samples_exist(tags_inf, tags_test):
    return [sample_fn(ti, tt).exists() for ti, tt in zip(tags_inf, tags_test)]


def _ensemble_samples_for(statistics, mask, *, test_mode="shame", tag_mock=TAG_MOCK_SHAME, n_total=N_ENSEMBLE_DRAWS):
    """Equal-weight K-member mixture + param_names from member 0."""
    samples, missing = load_ensemble_member_samples(
        statistics, mask,
        test_mode=test_mode, tag_mock=tag_mock,
        k_members=N_ENSEMBLE_K, n_total=n_total,
    )
    if samples is None:
        return None, None, "; ".join(missing)
    fn_pn = model_dir(statistics, mask, 0) / "param_names.txt"
    if not fn_pn.is_file():
        return None, None, f"missing param_names: {fn_pn}"
    return samples, list(np.loadtxt(fn_pn, dtype=str)), None


def ensemble_samples_exist(*, test_mode="shame", tag_mock=TAG_MOCK_SHAME):
    """True per fiducial combo if all K member sample files exist for this test."""
    keep = []
    for statistics, mask in zip(STATISTICS_ARR_FID, TAGS_MASK_FID):
        ok = True
        for nth in range(N_ENSEMBLE_K):
            path = member_samples_path(
                statistics, mask, nth, test_mode=test_mode, tag_mock=tag_mock,
            )
            if not path.is_file():
                ok = False
                break
        keep.append(ok)
    return keep


def load_fiducial_ensemble_samples_list(*, test_mode="shame", tag_mock=TAG_MOCK_SHAME):
    """Build samples_list / labels / colors / tags for plot_contours_inf."""
    samples_list, labels, colors, tags_inf, keep_idx = [], [], [], [], []
    for i, ((statistics, mask), color) in enumerate(zip(KP035_CONFIGS, COLORS_FID)):
        samples, param_names, err = _ensemble_samples_for(
            statistics, mask, test_mode=test_mode, tag_mock=tag_mock,
        )
        if samples is None:
            print(f"  ensemble missing [{get_stat_label_short(statistics)}]: {err}")
            continue
        if samples.ndim == 3:
            samples = samples[:, 0, :]
        samples_list.append((samples, list(param_names)))
        labels.append(get_stat_label_short(statistics))
        colors.append(color)
        tags_inf.append(f"_ens_top{N_ENSEMBLE_K}_{'_'.join(statistics)}{mask}")
        keep_idx.append(i)
    return samples_list, labels, colors, tags_inf, keep_idx


def setup_fixed_mean_test():
    tags_inf, labels, colors = setup_fiducial_inference()
    cosmo_vary, bias_vary, param_vary = utils_plot.load_training_params(
        TAG_PARAMS_TRAIN, TAG_BIASPARAMS_TRAIN, bx=bx,
    )
    tag_stats_arr = [f'_{"_".join(s)}' for s in STATISTICS_ARR_FID]
    tags_test = utils_plot.setup_test_tags(
        data_mode=data_mode,
        tag_params_test=TAG_PARAMS_TEST_FIXED,
        tags_biasparams_test=TAG_BIASPARAMS_TEST_FIXED,
        tag_stats_arr=tag_stats_arr,
        tag_noise_test=TAG_NOISE_TEST_FIXED,
        tag_datagen_test=TAG_DATAGEN_TEST_MEAN,
        tags_mask_test=TAGS_MASK_FID,
    )
    keep = samples_exist(tags_inf, tags_test)
    theta = data_loader.load_theta_test(
        TAG_PARAMS_TEST_FIXED,
        TAG_BIASPARAMS_TEST_FIXED,
        cosmo_param_names_vary=cosmo_vary,
        bias_param_names_vary=bias_vary,
    )
    theta_obs = theta[idx_obs] if theta.ndim == 2 else theta
    idxs_zoom = [param_vary.index(pn) for pn in PARAM_NAMES_ZOOM]
    return tags_inf, labels, colors, tags_test, keep, theta_obs, theta_obs[idxs_zoom], param_vary


def setup_coverage_test():
    tags_inf, labels, colors = setup_fiducial_inference()
    cosmo_vary, bias_vary, param_vary = utils_plot.load_training_params(
        TAG_PARAMS_TRAIN, TAG_BIASPARAMS_TRAIN, bx=bx,
    )
    tag_stats_arr = [f'_{"_".join(s)}' for s in STATISTICS_ARR_FID]
    tags_test = utils_plot.setup_test_tags(
        data_mode=data_mode,
        tag_params_test=TAG_PARAMS_TEST_COV,
        tags_biasparams_test=TAG_BIASPARAMS_TEST_COV,
        tag_stats_arr=tag_stats_arr,
        tag_noise_test=TAG_NOISE_TEST_COV,
        tag_datagen_test="",
        tags_mask_test=TAGS_MASK_FID,
    )
    keep = samples_exist(tags_inf, tags_test)
    return tags_inf, labels, colors, tags_test, keep, cosmo_vary, bias_vary, param_vary


def setup_shame_test():
    tags_inf, labels, colors = setup_fiducial_inference()
    cosmo_vary, bias_vary, param_vary = utils_plot.load_training_params(
        TAG_PARAMS_TRAIN, TAG_BIASPARAMS_TRAIN, bx=bx,
    )
    tag_stats_arr = [f'_{"_".join(s)}' + m for s, m in zip(STATISTICS_ARR_FID, TAGS_MASK_FID)]
    tags_test = utils_plot.setup_shame_mock_test_tags(
        tag_stats_arr=tag_stats_arr,
        data_mode_test=DATA_MODE_TEST_SHAME,
        tag_mock=TAG_MOCK_SHAME,
    )
    keep = samples_exist(tags_inf, tags_test)
    theta_dict = data_loader.load_params_ood(DATA_MODE_TEST_SHAME, TAG_MOCK_SHAME)
    theta_obs_full = np.array([theta_dict.get(pn, np.nan) for pn in param_vary])
    idxs_zoom = [param_vary.index(pn) for pn in PARAM_NAMES_ZOOM]
    return tags_inf, labels, colors, tags_test, keep, param_vary, theta_obs_full[idxs_zoom]


def plot_fiducial_contours(
    tags_inf, labels, colors, tags_test, keep, theta_show, *,
    extents=None, add_truth=True, save_name=None,
    param_names=None, unreparameterize=True,
    fontsize_legend=FONTSIZE_LEGEND,
    legend_location=None,
    legend_loc=None,
    loc_legend=LOC_LEGEND,
    samples_list=None,
    title=None,
):
    """Contour overlay. If samples_list is given, tags are only labels (ensemble path)."""
    if samples_list is not None:
        n = len(samples_list)
        if n == 0:
            print("No samples on disk for fiducial contours")
            return
        fig = plotter.plot_contours_inf(
            param_names=param_names or PARAM_NAMES_ZOOM,
            idx_obs=idx_obs,
            theta_obs_true=theta_show,
            inf_methods=["sbi"] * n,
            tags_inf=tags_inf,
            tags_test=[""] * n,
            colors=colors,
            labels=labels,
            title=title,
            extents=extents or extents_mid,
            figsize=FIGSIZE_CORNER,
            fontsize_legend=fontsize_legend,
            legend_location=legend_location,
            legend_loc=legend_loc,
            loc_legend=loc_legend,
            unreparameterize=unreparameterize,
            add_truth=add_truth,
            samples_list=samples_list,
            show=False,
        )
    else:
        idxs = [i for i, k in enumerate(keep) if k]
        if not idxs:
            print("No samples on disk for fiducial contours")
            for i, k in enumerate(keep):
                if not k:
                    print(f"  missing: {sample_fn(tags_inf[i], tags_test[i])}")
            return
        fig = plotter.plot_contours_inf(
            param_names=param_names or PARAM_NAMES_ZOOM,
            idx_obs=idx_obs,
            theta_obs_true=theta_show,
            inf_methods=["sbi"] * len(idxs),
            tags_inf=[tags_inf[i] for i in idxs],
            tags_test=[tags_test[i] for i in idxs],
            colors=[colors[i] for i in idxs],
            labels=[labels[i] for i in idxs],
            title=title,
            extents=extents or extents_mid,
            figsize=FIGSIZE_CORNER,
            fontsize_legend=fontsize_legend,
            legend_location=legend_location,
            legend_loc=legend_loc,
            loc_legend=loc_legend,
            unreparameterize=unreparameterize,
            add_truth=add_truth,
            show=False,
        )
    if save_name is not None and fig is not None:
        save_figure(fig, save_name)
    plt.show()


def plot_fiducial_ensemble_contours(
    *, test_mode="cvmean", tag_mock=TAG_MOCK_SHAME, theta_show=None,
    extents=None, save_name=None, param_names=None, unreparameterize=True,
    fontsize_legend=FONTSIZE_LEGEND, legend_location=None, legend_loc=None,
    loc_legend=LOC_LEGEND, title=None,
):
    """Multi-stat corner from equal-weight K=3 ensemble mixtures."""
    samples_list, labels, colors, tags_inf, _ = load_fiducial_ensemble_samples_list(
        test_mode=test_mode, tag_mock=tag_mock,
    )
    if not samples_list:
        print(f"No ensemble samples for fiducial contours ({test_mode})")
        return
    plot_fiducial_contours(
        tags_inf, labels, colors, [""] * len(tags_inf), [True] * len(tags_inf),
        theta_show, extents=extents, save_name=save_name, param_names=param_names,
        unreparameterize=unreparameterize, fontsize_legend=fontsize_legend,
        legend_location=legend_location, legend_loc=legend_loc, loc_legend=loc_legend,
        samples_list=samples_list, title=title,
    )


def stats_summary(arr):
    return (
        np.nanmedian(arr, axis=0),
        np.nanpercentile(arr, 1, axis=0),
        np.nanpercentile(arr, 16, axis=0),
        np.nanpercentile(arr, 84, axis=0),
        np.nanpercentile(arr, 99, axis=0),
    )


def plot_stat_band(ax, x, median, p1, p16, p84, p99, color, *, with_labels=True):
    # Median first for legend order; zorder keeps the line above the bands.
    lab_med = "training set median" if with_labels else None
    lab_1684 = "16–84%" if with_labels else None
    lab_199 = "1–99%" if with_labels else None
    ax.plot(x, median, color=color, label=lab_med, zorder=3)
    ax.fill_between(x, p16, p84, color=color, alpha=0.35, label=lab_1684, zorder=2)
    ax.fill_between(x, p1, p99, color=color, alpha=0.18, label=lab_199, zorder=1)


def filter_finite_coverage_rows(theta_true_arr, theta_pred_arr, covs_pred_arr):
    """Drop coverage rows with NaN predictions (failed / timed-out inference batches)."""
    theta_true_out, theta_pred_out, covs_out = [], [], []
    for i in range(theta_pred_arr.shape[0]):
        ok = np.all(np.isfinite(theta_pred_arr[i]), axis=1)
        if covs_pred_arr.ndim == 4:
            ok &= np.all(np.isfinite(covs_pred_arr[i]).all(axis=2), axis=1)
        theta_true_out.append(theta_true_arr[i][ok])
        theta_pred_out.append(theta_pred_arr[i][ok])
        covs_out.append(covs_pred_arr[i][ok])
    return np.array(theta_true_out), np.array(theta_pred_out), np.stack(covs_out, axis=0)


def _nbar_shades(hex_base):
    """Light → mid → dark for increasing n̄ (blend toward white / base / black)."""
    rgb = np.array(mcolors.to_rgb(hex_base))
    return [
        mcolors.to_hex(0.42 * rgb + 0.58 * np.ones(3)),
        mcolors.to_hex(rgb),
        mcolors.to_hex(0.42 * rgb + 0.58 * np.zeros(3)),
    ]


def fob_chi2_sqrt_limit(ndims):
    return float(np.sqrt(chi2.ppf(_CHI2_PPF_1SIG, int(ndims))))


def compute_fob(
    theta_pred_row, cov, param_names, theta_true, param_vary,
    *,
    param_names_fob=None,
    fob_keys=None,
):
    """FoB in (ω_c, σ8, b1) physical space by default, or (ω_c, σ8, σ8×b1) reparam space."""
    param_names_fob = list(param_names_fob or PARAM_NAMES_FOB)
    fob_keys = list(fob_keys or FOB_KEYS)
    names = list(param_names)
    tp = np.asarray(theta_pred_row, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    theta_true = np.asarray(theta_true, dtype=float)
    use_s8xb1 = "sigma8_cold_x_b1" in param_names_fob

    if use_s8xb1:
        # FoB in NPE training coordinates (already reparametrized samples).
        if not ui.has_reparameterized_sigma8_columns(names):
            raise ValueError(
                "FoB(σ8×b1) requires reparameterized samples with sigma8_cold_x_b1"
            )
        theta_true_rep, names_rep = ui.reparameterize_theta(theta_true, param_vary)
        names_rep = list(names_rep)
        true_vec = np.array(
            [float(theta_true_rep[names_rep.index(pn)]) for pn in names], dtype=float,
        )
        true_fob = np.array(
            [true_vec[names.index(pn)] for pn in param_names_fob], dtype=float,
        )
        mu = np.array(
            [float(tp[names.index(pn)]) for pn in param_names_fob], dtype=float,
        )
        idx_fob = [names.index(pn) for pn in param_names_fob]
    else:
        # FoB in physical (ω_c, σ8, b1); unreparameterize if needed.
        if ui.has_reparameterized_sigma8_columns(names):
            theta_true_rep, names_rep = ui.reparameterize_theta(theta_true, param_vary)
            names_rep = list(names_rep)
            true_aligned = np.array(
                [float(theta_true_rep[names_rep.index(pn)]) for pn in names], dtype=float,
            )
            tp_u, tt_u, cov_u, names_u = ui.unreparameterize_prediction_block(
                tp[None, :], true_aligned[None, :], names, covs_per_sample=cov_arr[None, :, :],
            )
            tp, true_vec, cov_arr, names = tp_u[0], tt_u[0], cov_u[0], list(names_u)
        else:
            names_phys = list(param_vary)
            true_vec = np.array(
                [float(theta_true[names_phys.index(pn)]) for pn in names], dtype=float,
            )

        true_fob = np.array([true_vec[names.index(pn)] for pn in param_names_fob], dtype=float)
        mu = np.array([float(tp[names.index(pn)]) for pn in param_names_fob], dtype=float)
        idx_fob = [names.index(pn) for pn in param_names_fob]

    cov_sub = cov_arr[np.ix_(idx_fob, idx_fob)] + FOB_RIDGE * np.eye(len(idx_fob))
    fob = {}
    for key, i in zip(fob_keys, range(len(fob_keys))):
        sig = np.sqrt(cov_sub[i, i])
        fob[key] = abs(mu[i] - true_fob[i]) / sig if sig > 0 else np.nan
    diff = mu - true_fob
    sign, logdet = np.linalg.slogdet(cov_sub)
    cov_inv = np.linalg.inv(cov_sub) if sign > 0 else np.linalg.pinv(cov_sub)
    fob3 = float(np.sqrt(diff @ cov_inv @ diff))
    return fob, fob3


def compute_fom_key_block(cov, param_names, theta_pred_row=None):
    """FoM in physical (ω_c, σ8, b1) space; unreparameterizes cov if needed."""
    names = list(param_names)
    cov_arr = np.asarray(cov, dtype=float)
    if ui.has_reparameterized_sigma8_columns(names):
        if theta_pred_row is None:
            raise ValueError("theta_pred_row required to unreparameterize FoM covariance")
        tp = np.asarray(theta_pred_row, dtype=float)
        sigma8 = float(tp[names.index("sigma8_cold")])
        cov_arr = ui.scale_covariance_unreparameterize_approx(cov_arr, names, sigma8)
        _, names = ui.unreparameterize_theta(tp, names)
        names = list(names)
    idx_fob = [names.index(pn) for pn in PARAM_NAMES_FOB]
    cov_sub = cov_arr[np.ix_(idx_fob, idx_fob)]
    fom_marg = {}
    for key, i in zip(FOB_KEYS, range(len(FOB_KEYS))):
        sig = np.sqrt(cov_sub[i, i])
        fom_marg[key] = 1.0 / sig if sig > 0 else np.nan
    det = np.linalg.det(cov_sub)
    fom_3d = 1.0 / np.sqrt(det) if det > 0 else np.nan
    return fom_marg, float(fom_3d)


def compute_fom_full(cov):
    """FoM = 1/sqrt(det C) over all parameters in the chain covariance."""
    return float(ui.figure_of_merit(np.asarray(cov, dtype=float)))


# --- Scale / FoB helpers ---

def kmax_kb_numeric(tag_kb):
    if tag_kb == "":
        return KMAX_PK
    return float(tag_kb.replace("_kb", ""))


def kmax_kpgm_numeric(tag_kpgm):
    if tag_kpgm == "":
        return KMAX_PK
    return float(tag_kpgm.replace("_kpgm", ""))


def _with_fixed_kp(mask_suffix):
    """Prepend fiducial _kp0.35 to kb/kpgm (or empty) suffixes.

    Note: ``_kpgm*`` also starts with ``_kp`` as a string prefix — match ``_kp``
    only when it is the Pk cut tag (``_kp0.`` / ``_kp`` alone), not ``_kpgm``.
    """
    if mask_suffix == "_kp" or mask_suffix.startswith("_kp0") or mask_suffix.startswith("_kp1"):
        return mask_suffix
    return f"_kp{KMAX_PK:g}{mask_suffix}"


def _mask_candidates_kb(tag_kb):
    # pk+bispec: joined _kp0.35_kb*
    return [_with_fixed_kp(tag_kb)]


def _joined_kb_kpgm(tag_kb, tag_kpgm):
    return _with_fixed_kp(f"{tag_kb}{tag_kpgm}")


def _joint_mask_candidates(tag_kb, tag_kpgm):
    return [_joined_kb_kpgm(tag_kb, tag_kpgm)]


def _shame_samples_path(statistics_row, tag_mask, tag_mock=TAG_MOCK_SHAME):
    """K=3 ensemble mixture for SHAMe OOD (equal-weight)."""
    return _ensemble_samples_for(
        statistics_row, tag_mask, test_mode="shame", tag_mock=tag_mock,
    )


def load_shame_fob3(
    statistics_row, tag_mask, tag_mock=TAG_MOCK_SHAME,
    *, param_names_fob=None, fob_keys=None,
):
    """Return (fob3, None) on success, or (None, reason) if missing/invalid."""
    if param_names_fob is None:
        param_names_fob = getattr(load_shame_fob3, "_param_names_fob", None)
    if fob_keys is None:
        fob_keys = getattr(load_shame_fob3, "_fob_keys", None)
    cosmo_vary, bias_vary, param_vary = utils_plot.load_training_params(
        TAG_PARAMS_TRAIN, TAG_BIASPARAMS_TRAIN, bx=bx,
    )
    samples, param_names, path_err = _shame_samples_path(
        statistics_row, tag_mask, tag_mock=tag_mock,
    )
    if samples is None:
        return None, path_err
    # Timeout placeholders are all-NaN finals — treat as missing so the
    # scale sweep does not silently drop points below the first finite k_max.
    if not np.isfinite(samples).any():
        return None, f"all-NaN samples (mask={tag_mask!r}, mock={tag_mock!r})"
    if samples.ndim == 2:
        samples = samples[:, np.newaxis, :]
    theta_pred = np.mean(samples[:, 0, :], axis=0)
    cov = np.cov(samples[:, 0, :].T)
    theta_phys = data_loader.load_theta_ood(
        DATA_MODE_TEST_SHAME,
        tag_mock,
        cosmo_param_names_vary=cosmo_vary,
        bias_param_names_vary=bias_vary,
    )
    _, fob3 = compute_fob(
        theta_pred, cov, param_names, theta_phys, param_vary,
        param_names_fob=param_names_fob, fob_keys=fob_keys,
    )
    if fob3 is None or not np.isfinite(fob3):
        return None, f"non-finite FoB3 (mask={tag_mask!r}, mock={tag_mock!r})"
    return fob3, None


def _try_load_shame_fob3(
    statistics_row, mask_candidates, tag_mock=TAG_MOCK_SHAME,
    *, param_names_fob=None, fob_keys=None,
):
    reasons = []
    for tag_mask in mask_candidates:
        val, reason = load_shame_fob3(
            statistics_row, tag_mask, tag_mock=tag_mock,
            param_names_fob=param_names_fob, fob_keys=fob_keys,
        )
        if val is not None:
            return val, None
        reasons.append(f"{tag_mask!r}: {reason}")
    return None, reasons


def _report_missing_fob3(panel_title, statistics_row, tag_mock, kmax, reasons):
    label = get_stat_label_short(statistics_row)
    detail = "; ".join(reasons) if reasons else "unknown"
    print(
        f"MISSING FoB3 point | panel={panel_title} | stats={label} | "
        f"nbar={tag_mock} | k_max={kmax} | tried: {detail}"
    )


def _sweep_bispec_fob3(
    statistics_row, *, tag_kpgm_fixed="_kpgm0.25", tag_mock=TAG_MOCK_SHAME, report_missing=False,
):
    xs, ys = [], []
    panel_title = "bispec sweep (PGM kmax=0.25 fixed)"
    for tag_kb in TAGS_KMAX_KB:
        if "bispec" in statistics_row:
            if "pgm" in statistics_row:
                candidates = _joint_mask_candidates(tag_kb, tag_kpgm_fixed)
            else:
                candidates = _mask_candidates_kb(tag_kb)
        else:
            candidates = [_with_fixed_kp(tag_kpgm_fixed)]
        val, reasons = _try_load_shame_fob3(
            statistics_row, candidates, tag_mock=tag_mock,
        )
        kmax = kmax_kb_numeric(tag_kb)
        if val is None:
            if report_missing:
                _report_missing_fob3(panel_title, statistics_row, tag_mock, kmax, reasons)
            continue
        xs.append(kmax)
        ys.append(val)
    return np.array(xs), np.array(ys)


def _sweep_pgm_fob3(
    statistics_row, *, tag_kb_fixed="_kb0.25", tag_mock=TAG_MOCK_SHAME, report_missing=False,
):
    xs, ys = [], []
    panel_title = "PGM sweep (bispec kmax=0.25 fixed)"
    for tag_kpgm in TAGS_KMAX_KPGM:
        if "pgm" in statistics_row:
            if "bispec" in statistics_row:
                candidates = _joint_mask_candidates(tag_kb_fixed, tag_kpgm)
            else:
                candidates = [_with_fixed_kp(tag_kpgm)]
        else:
            candidates = _mask_candidates_kb(tag_kb_fixed)
        val, reasons = _try_load_shame_fob3(
            statistics_row, candidates, tag_mock=tag_mock,
        )
        kmax = kmax_kpgm_numeric(tag_kpgm)
        if val is None:
            if report_missing:
                _report_missing_fob3(panel_title, statistics_row, tag_mock, kmax, reasons)
            continue
        xs.append(kmax)
        ys.append(val)
    return np.array(xs), np.array(ys)


def _constant_fob3_on_sweep(
    statistics_row, *, bispec_sweep, tag_mock=TAG_MOCK_SHAME, report_missing=False,
):
    """FoB3 for combos that do not vary along the swept k_max axis."""
    if bispec_sweep and statistics_row == ["pk", "pgm"]:
        candidates = ["_kp0.35_kpgm0.25"]
        panel_title = "bispec sweep (constant Pgg+Pgm)"
        kmax = 0.25
    elif not bispec_sweep and statistics_row == ["pk", "bispec"]:
        candidates = ["_kp0.35_kb0.25"]
        panel_title = "PGM sweep (constant Pgg+Bggg)"
        kmax = 0.25
    else:
        return None
    val, reasons = _try_load_shame_fob3(
        statistics_row, candidates, tag_mock=tag_mock,
    )
    if val is None and report_missing:
        _report_missing_fob3(panel_title, statistics_row, tag_mock, kmax, reasons)
    return val


def kmax_label_to_upper_edge(kmax_label, *, family="pk"):
    """Upper edge of the last included bin for a nominal kmax mask label."""
    kmax = float(kmax_label)
    if family in ("pk", "pgm"):
        centers, edges = K_CENTERS_PK, K_EDGES_PK
    else:
        raise ValueError(f"unknown family={family!r}")
    included = np.where(centers < kmax)[0]
    if included.size == 0:
        return float(edges[0])
    return float(edges[int(included[-1]) + 1])


def kmax_plot_x_pk(kmax_label):
    return kmax_label_to_upper_edge(kmax_label, family="pk")


def overall_k_mask(statistics_row, k):
    """Joined tags_mask for joint overall scale cut at k.

    For k <= 0.25 every statistic uses k; for k > 0.25, Pk uses k while
    bispec and Pgm are capped at 0.25.
    """
    k_other = k if k <= 0.25 else 0.25
    parts = []
    for s in statistics_row:
        if s == "pk":
            parts.append(f"_kp{k}")
        elif s == "bispec":
            parts.append(f"_kb{k_other}")
        elif s == "pgm":
            parts.append(f"_kpgm{k_other}")
        else:
            raise ValueError(s)
    return "".join(parts)


def load_ensemble_fom3(statistics_row, tag_mask, *, test_mode="shame", tag_mock=TAG_MOCK_SHAME):
    samples, param_names, path_err = _ensemble_samples_for(
        statistics_row, tag_mask, test_mode=test_mode, tag_mock=tag_mock,
    )
    if samples is None:
        return None, path_err
    if samples.ndim == 2:
        samples = samples[:, np.newaxis, :]
    theta_pred = np.mean(samples[:, 0, :], axis=0)
    cov = np.cov(samples[:, 0, :].T)
    _, fom3 = compute_fom_key_block(cov, param_names, theta_pred_row=theta_pred)
    if fom3 is None or not np.isfinite(fom3):
        return None, f"non-finite FoM3 (mask={tag_mask!r}, mode={test_mode})"
    return fom3, None


def load_ensemble_fom_marg(
    statistics_row, tag_mask, param_key, *, test_mode="shame", tag_mock=TAG_MOCK_SHAME,
):
    """Marginal FoM for one key in FOB_KEYS (omega_m / sigma8 / b1)."""
    samples, param_names, path_err = _ensemble_samples_for(
        statistics_row, tag_mask, test_mode=test_mode, tag_mock=tag_mock,
    )
    if samples is None:
        return None, path_err
    if samples.ndim == 2:
        samples = samples[:, np.newaxis, :]
    if not np.isfinite(samples).any():
        return None, f"all-NaN samples (mask={tag_mask!r}, mode={test_mode})"
    theta_pred = np.mean(samples[:, 0, :], axis=0)
    cov = np.cov(samples[:, 0, :].T)
    fom_marg, _ = compute_fom_key_block(cov, param_names, theta_pred_row=theta_pred)
    val = fom_marg.get(param_key)
    if val is None or not np.isfinite(val):
        return None, f"non-finite FoM({param_key!r}) (mask={tag_mask!r}, mode={test_mode})"
    return float(val), None


def load_ensemble_fom_full(statistics_row, tag_mask, *, test_mode="shame", tag_mock=TAG_MOCK_SHAME):
    samples, param_names, path_err = _ensemble_samples_for(
        statistics_row, tag_mask, test_mode=test_mode, tag_mock=tag_mock,
    )
    if samples is None:
        return None, path_err
    if samples.ndim == 2:
        samples = samples[:, np.newaxis, :]
    cov = np.cov(samples[:, 0, :].T)
    fom = compute_fom_full(cov)
    if fom is None or not np.isfinite(fom):
        return None, f"non-finite FoM_full (mask={tag_mask!r}, mode={test_mode})"
    return float(fom), None


def _overall_kmax_fom_marg_ensemble(
    statistics_row, param_key, *, test_mode="shame", tag_mock=TAG_MOCK_SHAME, report_missing=False,
):
    xs, ys = [], []
    for k in K_OVERALL:
        tag_mask = overall_k_mask(statistics_row, k)
        val, reason = load_ensemble_fom_marg(
            statistics_row, tag_mask, param_key, test_mode=test_mode, tag_mock=tag_mock,
        )
        if val is None:
            if report_missing:
                print(
                    f"MISSING FoM | overall kmax ensemble | param={param_key} | "
                    f"stats={get_stat_label_short(statistics_row)} | {tag_mock} | k={k} | {reason}"
                )
            continue
        xs.append(kmax_plot_x_pk(k))
        ys.append(val)
    return np.array(xs), np.array(ys)


def _overall_kmax_fom_ensemble(
    statistics_row, *, kind="3d", test_mode="shame", tag_mock=TAG_MOCK_SHAME, report_missing=False,
):
    xs, ys = [], []
    loader = load_ensemble_fom3 if kind == "3d" else load_ensemble_fom_full
    for k in K_OVERALL:
        tag_mask = overall_k_mask(statistics_row, k)
        val, reason = loader(
            statistics_row, tag_mask, test_mode=test_mode, tag_mock=tag_mock,
        )
        if val is None:
            if report_missing:
                print(
                    f"MISSING FoM-{kind} | overall kmax ensemble | "
                    f"stats={get_stat_label_short(statistics_row)} | {tag_mock} | k={k} | {reason}"
                )
            continue
        xs.append(kmax_plot_x_pk(k))
        ys.append(val)
    return np.array(xs), np.array(ys)


# --- Convergence helpers ---

def build_tag_inf_conv(bx_val, n_cosmo_val, statistics, tag_masks, *, nth=0):
    tag_stats = f"_{'_'.join(statistics)}"
    base = (
        f"_{data_mode}{tag_stats}{tag_masks}{TAG_PARAMS_TRAIN}"
        f"{TAG_BIASPARAMS_TRAIN}{TAG_NOISE_TRAIN}{tag_reparam}"
        f"_bx{bx_val}_ntrain{n_cosmo_val}{TAG_INF_BEST_SUFFIX}"
    )
    if nth == 0:
        return base
    return f"{base}_nbest{nth}"


def build_tag_test_conv(statistics, tag_masks):
    tag_stats = f"_{'_'.join(statistics)}"
    return (
        f"_{data_mode}{tag_stats}{tag_masks}"
        f"{TAG_PARAMS_TEST_COV}{TAG_BIASPARAMS_TEST_COV}{TAG_NOISE_TEST_COV}"
    )


def build_tag_test_cv(statistics, tag_masks, *, mean=False):
    """Fixed-cosmo shame test tag (cosmic variance). mean=True → evaluate_mean (_mean)."""
    tag_stats = f"_{'_'.join(statistics)}"
    tag_mean = TAG_DATAGEN_TEST_MEAN if mean else ""
    return (
        f"_{data_mode}{tag_stats}{tag_masks}"
        f"{TAG_PARAMS_TEST_FIXED}{TAG_BIASPARAMS_TEST_FIXED}{TAG_NOISE_TEST_FIXED}"
        f"{tag_mean}"
    )


def _coverage_samples_path(tag_inf, tag_test):
    """Return (path, kind) with kind in {'done', 'inprogress'}, or (None, None)."""
    base = dir_sbi / f"sbi{tag_inf}"
    fn = base / f"samples_test{tag_test}_pred.npy"
    if fn.is_file():
        return fn, "done"
    fn_ip = base / f"samples_test{tag_test}_pred_inprogress.npy"
    if fn_ip.is_file():
        return fn_ip, "inprogress"
    return None, None


def _load_ensemble_coverage_3d(statistics, tag_masks, *, bx_val=None, n_cosmo_val=None):
    """Load K=3 coverage chains concatenated on the draw axis.

    For the fiducial (bx, n_train) grid point use ``member_samples_path``.
    For arbitrary (bx, n_cosmo) try ``_best-rand30`` + ``_nbest1/2`` under the
    same tag_test; fall back to top-1 only if not all K exist.
    Returns (samples_3d, param_names, info_kind) or (None, None, reason).
    """
    if bx_val is None:
        bx_val = bx
    if n_cosmo_val is None:
        n_cosmo_val = n_train
    tag_test = build_tag_test_conv(statistics, tag_masks)
    arrays = []
    kinds = []
    for nth in range(N_ENSEMBLE_K):
        tag_inf = build_tag_inf_conv(bx_val, n_cosmo_val, statistics, tag_masks, nth=nth)
        fn, kind = _coverage_samples_path(tag_inf, tag_test)
        if fn is None:
            break
        arr = np.load(fn)
        if arr.ndim == 2:
            arr = arr[:, np.newaxis, :]
        if not np.isfinite(arr).any():
            break
        arrays.append(arr)
        kinds.append(kind)
    if len(arrays) == N_ENSEMBLE_K:
        # Align n_obs to the minimum across members (inprogress may differ).
        n_obs = min(a.shape[1] for a in arrays)
        arrays = [a[:, :n_obs, :] for a in arrays]
        mixed = np.concatenate(arrays, axis=0)
        tag_inf0 = build_tag_inf_conv(bx_val, n_cosmo_val, statistics, tag_masks, nth=0)
        fn_pn = dir_sbi / f"sbi{tag_inf0}" / "param_names.txt"
        names = list(np.loadtxt(fn_pn, dtype=str)) if fn_pn.is_file() else None
        kind = "ensemble_inprogress" if "inprogress" in kinds else "ensemble_done"
        return mixed, names, kind
    # Fall back to top-1.
    tag_inf = build_tag_inf_conv(bx_val, n_cosmo_val, statistics, tag_masks, nth=0)
    fn, kind = _coverage_samples_path(tag_inf, tag_test)
    if fn is None:
        return None, None, f"no coverage samples for {tag_inf}"
    arr = np.load(fn)
    if arr.ndim == 2:
        arr = arr[:, np.newaxis, :]
    fn_pn = dir_sbi / f"sbi{tag_inf}" / "param_names.txt"
    names = list(np.loadtxt(fn_pn, dtype=str)) if fn_pn.is_file() else None
    return arr, names, kind or "done"


def _finite_sample_rows(samples_arr):
    """Indices of coverage rows with finite posterior chains (failed batches are all-NaN)."""
    if samples_arr.ndim != 3:
        raise ValueError(f"expected 3D samples, got {samples_arr.shape}")
    # Skip all-NaN obs before nanmean to avoid "Mean of empty slice" warnings.
    has_any = np.any(np.isfinite(samples_arr), axis=(0, 2))
    n_obs, n_par = samples_arr.shape[1], samples_arr.shape[2]
    theta_pred = np.full((n_obs, n_par), np.nan, dtype=float)
    if np.any(has_any):
        theta_pred[has_any] = np.nanmean(samples_arr[:, has_any, :], axis=0)
    return np.where(np.all(np.isfinite(theta_pred), axis=1))[0]


def _moments_from_samples_3d(samples_arr, param_names):
    rows = _finite_sample_rows(samples_arr)
    npar = samples_arr.shape[2]
    if rows.size == 0:
        return np.empty((0, npar)), np.empty((0, npar, npar)), list(param_names), rows
    theta_pred = np.array([np.nanmean(samples_arr[:, i, :], axis=0) for i in rows])
    covs_pred = np.array([np.cov(samples_arr[:, i, :].T) for i in rows])
    return theta_pred, covs_pred, list(param_names), rows


def load_coverage_metrics(
    bx, n_cosmo, statistics, tag_masks, tag_test=None, *,
    tag_params_test=None, tag_biasparams_test=None, n_target=None,
):
    """
    Load FoM/FoB / MSE / delta metrics for one grid point.

    tag_params_test / tag_biasparams_test select the truth vector
    (coverage LH by default; fixed-cosmo shame for CV).
    n_target: expected usable obs (N_COVERAGE for coverage/CV-sample; 1 for CV-mean).

    Returns
    -------
    row : dict or None
    info : dict with keys status, message, path, n_stored, n_cov, kind
    """
    if tag_test is None:
        tag_test = build_tag_test_conv(statistics, tag_masks)
    if tag_params_test is None:
        tag_params_test = TAG_PARAMS_TEST_COV
    if tag_biasparams_test is None:
        tag_biasparams_test = TAG_BIASPARAMS_TEST_COV
    if n_target is None:
        n_target = N_COVERAGE
    tag_inf = build_tag_inf_conv(bx, n_cosmo, statistics, tag_masks)
    label = utils_plot.get_stat_label(statistics)
    base_info = {
        "label": label, "bx": bx, "n_cosmo": n_cosmo,
        "tag_inf": tag_inf, "path": None, "kind": None,
        "n_stored": 0, "n_cov": 0, "status": "missing", "message": "",
    }

    model_dir_path = dir_sbi / f"sbi{tag_inf}"
    if not model_dir_path.is_dir():
        base_info["message"] = f"no model directory: {model_dir_path.name}"
        return None, base_info

    # Prefer K=3 ensemble mix for the standard coverage test tag.
    use_ensemble = tag_test == build_tag_test_conv(statistics, tag_masks)
    if use_ensemble:
        samples_arr, param_names_all, kind = _load_ensemble_coverage_3d(
            statistics, tag_masks, bx_val=bx, n_cosmo_val=n_cosmo,
        )
        if samples_arr is None:
            base_info["message"] = str(kind)
            return None, base_info
        base_info["kind"] = kind
        base_info["path"] = f"ensemble×{N_ENSEMBLE_K}" if str(kind).startswith("ensemble") else str(kind)
        if param_names_all is None:
            base_info["status"] = "bad_model"
            base_info["message"] = "missing param_names.txt for ensemble coverage"
            return None, base_info
    else:
        fn, kind = _coverage_samples_path(tag_inf, tag_test)
        base_info["kind"] = kind
        if fn is None:
            base_info["message"] = (
                f"no coverage samples "
                f"(looked for samples_test…_pred.npy / _pred_inprogress.npy under {model_dir_path.name})"
            )
            return None, base_info
        base_info["path"] = str(fn)
        samples_arr = np.load(fn)
        fn_pn = model_dir_path / "param_names.txt"
        if not fn_pn.is_file():
            base_info["status"] = "bad_model"
            base_info["message"] = f"missing param_names.txt in {model_dir_path.name} (file={fn.name})"
            return None, base_info
        param_names_all = np.loadtxt(fn_pn, dtype=str)

    if samples_arr.ndim != 3:
        base_info["status"] = "bad_shape"
        base_info["message"] = (
            f"expected 3D samples (n_draw, n_obs, n_par), got shape={samples_arr.shape}"
        )
        return None, base_info

    n_stored = int(samples_arr.shape[1])
    base_info["n_stored"] = n_stored
    if n_stored == 0:
        base_info["status"] = "empty"
        base_info["message"] = "samples file has n_obs=0"
        return None, base_info

    theta_pred, covs_pred, param_names, rows = _moments_from_samples_3d(
        samples_arr, param_names_all,
    )
    n_use = int(rows.size)
    base_info["n_cov"] = n_use
    n_nan_rows = n_stored - n_use

    if n_use == 0:
        base_info["status"] = "all_nan"
        base_info["message"] = (
            f"loaded ({kind}): shape={samples_arr.shape}, "
            f"but all {n_stored} stored obs are all-NaN / unusable"
        )
        return None, base_info

    kind_str = str(kind)
    is_incomplete = (
        "inprogress" in kind_str or n_stored < n_target or n_use < n_target
    )
    if is_incomplete:
        status = "incomplete"
        parts = [f"loaded ({kind})", f"shape={tuple(samples_arr.shape)}"]
        parts.append(f"usable={n_use}/{n_target} target")
        if n_stored != n_use:
            parts.append(f"stored={n_stored} ({n_nan_rows} all-NaN/failed rows dropped)")
        if "inprogress" in kind_str:
            parts.append("still inprogress — final _pred.npy not written yet")
        elif n_stored < n_target:
            parts.append(f"file truncated: only {n_stored} obs stored")
        base_info["status"] = status
        base_info["message"] = "; ".join(parts)
    else:
        base_info["status"] = "complete"
        base_info["message"] = (
            f"loaded ({kind}): usable={n_use}/{n_target}"
            + (f" (dropped {n_nan_rows} failed rows)" if n_nan_rows else "")
        )

    cosmo_vary, bias_vary, param_vary = utils_plot.load_training_params(
        TAG_PARAMS_TRAIN, TAG_BIASPARAMS_TRAIN, bx=bx,
    )
    theta_true_all = data_loader.load_theta_test(
        tag_params_test, tag_biasparams_test,
        cosmo_param_names_vary=cosmo_vary, bias_param_names_vary=bias_vary,
    )
    if theta_true_all.ndim == 1:
        theta_true_all = np.tile(theta_true_all, (samples_arr.shape[1], 1))
    theta_true_all = theta_true_all[rows]

    fob_lists = {key: [] for key in FOB_KEYS}
    fom_marg_lists = {key: [] for key in FOB_KEYS}
    mse_lists = {key: [] for key in PARAMS_TRACK}
    delta_lists = {key: [] for key in PARAMS_TRACK}
    fob3_list = []
    fom_3d_list = []
    param_vary_list = list(param_vary)
    for i in range(n_use):
        fob, fob3 = compute_fob(theta_pred[i], covs_pred[i], param_names, theta_true_all[i], param_vary)
        fom_marg, fom_3d = compute_fom_key_block(
            covs_pred[i], param_names, theta_pred_row=theta_pred[i],
        )
        for key in FOB_KEYS:
            fob_lists[key].append(fob[key])
            fom_marg_lists[key].append(fom_marg[key])
        fob3_list.append(fob3)
        fom_3d_list.append(fom_3d)

        theta_pred_phys, names_phys = ui.unreparameterize_theta(theta_pred[i], param_names)
        names_phys = list(names_phys)
        for key, pname in PARAMS_TRACK.items():
            true_val = float(theta_true_all[i, param_vary_list.index(pname)])
            pred_val = float(theta_pred_phys[names_phys.index(pname)])
            err = pred_val - true_val
            mse_lists[key].append(err ** 2)
            delta_lists[key].append(err)

    row = {
        "bx": bx, "n_cosmo": n_cosmo, "n_sims": bx * n_cosmo,
        "n_cov": n_use, "n_stored": n_stored, "status": base_info["status"],
        "kind": kind,
    }
    for key in FOB_KEYS:
        row[f"fom_{key}"] = float(np.nanmean(fom_marg_lists[key]))
        row[f"fob_{key}"] = float(np.nanmean(fob_lists[key]))
    for key in PARAMS_TRACK:
        row[f"mse_{key}"] = float(np.nanmedian(mse_lists[key]))
        row[f"delta_{key}"] = float(np.nanmedian(delta_lists[key]))
    row["fom_3d"] = float(np.nanmean(fom_3d_list))
    row["fob3"] = float(np.nanmean(fob3_list))
    return row, base_info


def collect_coverage_grid(
    statistics, tag_masks, n_cosmo_values, bx_values, *, verbose=True,
    tag_test=None, tag_params_test=None, tag_biasparams_test=None, n_target=None,
    grid_label="Coverage",
):
    if tag_test is None:
        tag_test = build_tag_test_conv(statistics, tag_masks)
    label = utils_plot.get_stat_label(statistics)
    rows = []
    n_complete = n_incomplete = n_missing = 0
    if verbose:
        print(f"=== {grid_label} grid: {label} ===")
        print(f"tag_test = {tag_test}")
        print(f"grid: {len(bx_values)} bx × {len(n_cosmo_values)} n_cosmo = {len(bx_values) * len(n_cosmo_values)} points\n")

    for bx_val in bx_values:
        for n_cosmo in n_cosmo_values:
            m, info = load_coverage_metrics(
                bx_val, n_cosmo, statistics, tag_masks, tag_test=tag_test,
                tag_params_test=tag_params_test,
                tag_biasparams_test=tag_biasparams_test,
                n_target=n_target,
            )
            prefix = f"[{label}] bx={bx_val}, n_cosmo={n_cosmo}"
            if m is None:
                n_missing += 1
                if verbose:
                    print(f"MISSING  {prefix}: {info['message']}")
                continue
            rows.append(m)
            if info["status"] == "complete":
                n_complete += 1
                if verbose:
                    print(f"OK       {prefix}: {info['message']}")
            else:
                n_incomplete += 1
                if verbose:
                    print(f"INCOMPLETE {prefix}: {info['message']}")

    n_expected = len(bx_values) * len(n_cosmo_values)
    if verbose:
        print(
            f"\nSummary [{label}]: {len(rows)} loaded "
            f"({n_complete} complete, {n_incomplete} incomplete) / {n_expected} expected; "
            f"{n_missing} missing\n"
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def n_cosmo_alpha(n_cosmo, n_cosmo_values=None, lo=NCOSMO_ALPHA_LO, hi=NCOSMO_ALPHA_HI):
    n_cosmo_values = sorted(n_cosmo_values if n_cosmo_values is not None else n_cosmo_arr)
    if len(n_cosmo_values) == 1:
        return hi
    t = n_cosmo_values.index(int(n_cosmo)) / (len(n_cosmo_values) - 1)
    return lo + t * (hi - lo)


def _draw_nsims_curves(ax, df, y_col, color, *, n_cosmo_values=None, y_scale=1.0):
    """Plot N_train curves (6-11 paper style: solid lines, no markers)."""
    n_cosmo_values = sorted(n_cosmo_values if n_cosmo_values is not None else df["n_cosmo"].unique())
    rgb = mcolors.to_rgb(color)
    for n_cosmo in n_cosmo_values:
        grp = df[df["n_cosmo"] == n_cosmo].sort_values("n_sims")
        if grp.empty or y_col not in grp.columns:
            continue
        rgba = (*rgb, n_cosmo_alpha(n_cosmo, n_cosmo_values))
        if len(grp) >= 1:
            ax.plot(
                grp["n_sims"], grp[y_col] / y_scale,
                color=rgba, ls="-", lw=1.5, zorder=1,
            )


def _n_cosmo_legend_handles(n_cosmo_values):
    handles, labels = [], []
    for n_cosmo in sorted(n_cosmo_values):
        alpha = n_cosmo_alpha(n_cosmo, n_cosmo_values)
        handles.append(plt.Line2D([0], [0], color=(0.5, 0.5, 0.5, alpha), ls="-", lw=1.5))
        labels.append(rf"$N_\mathrm{{cosmo}}={n_cosmo}$")
    return handles, labels


def _conv_legend_outside(fig, stat_handles=None, stat_labels=None, n_cosmo_values=None,
                         *, fontsize=11, bbox_to_anchor=(0.91, 0.5), right=0.90):
    handles, labels = [], []
    if stat_handles is not None:
        handles.extend(stat_handles)
        labels.extend(stat_labels)
    if n_cosmo_values is not None:
        h_nc, l_nc = _n_cosmo_legend_handles(n_cosmo_values)
        handles.extend(h_nc)
        labels.extend(l_nc)
    fig.legend(
        handles, labels, fontsize=fontsize, loc="center left",
        bbox_to_anchor=bbox_to_anchor, frameon=False, labelcolor="black",
    )
    fig.subplots_adjust(right=right)


def plot_conv_fom3d_fob3_nsims(
    df,
    color,
    *,
    n_cosmo_values=None,
    save_name=None,
):
    if df.empty:
        print("No data for convergence plot")
        return
    n_cosmo_values = sorted(n_cosmo_values if n_cosmo_values is not None else df["n_cosmo"].unique())
    ref3 = fob_chi2_sqrt_limit(3)
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    _draw_nsims_curves(axes[0], df, "fom_3d", color, n_cosmo_values=n_cosmo_values)
    axes[0].set_ylabel(FOM3_YLABEL)
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    _draw_nsims_curves(
        axes[1], df, "fob3", color, n_cosmo_values=n_cosmo_values, y_scale=ref3,
    )
    h_ref = axes[1].axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7, label=r"$1\sigma$")
    axes[1].set_ylabel(FOB3_YLABEL)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(XLABEL_NSIMS)
    h_nc, l_nc = _n_cosmo_legend_handles(n_cosmo_values)
    axes[0].legend(h_nc, l_nc, fontsize=8, loc="best", frameon=False, labelcolor="black")
    axes[1].legend(handles=[h_ref], labels=[r"$1\sigma$"], fontsize=8, loc="best", frameon=False, labelcolor="black")
    fig.tight_layout()
    if save_name is not None:
        save_figure(fig, save_name)
    plt.show()


def plot_conv_all_combos_nsims(df_all, *, save_name=None):
    if df_all.empty:
        print("No data for convergence plot")
        return
    missing = [f"{p}_{k}" for k in FOB_KEYS for p in ("fom", "fob") if f"{p}_{k}" not in df_all.columns]
    if missing:
        raise KeyError(
            "df_all missing columns " + ", ".join(missing)
            + " (stale cache from sigma8_b1 era? rebuild with OVERWRITE_CACHE=True)"
        )
    full_labels = [utils_plot.get_stat_label(s) for s in STATISTICS_ARR_FID]
    short_labels = [get_stat_label_short(s) for s in STATISTICS_ARR_FID]
    n_cosmo_values = sorted(df_all["n_cosmo"].unique())
    gauss_ref = np.sqrt(2.0 / np.pi)
    lim1 = fob_chi2_sqrt_limit(1)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, key in zip(axes[0], FOB_KEYS):
        for full_label, color in zip(full_labels, COLORS_FID):
            sub = df_all[df_all["stat_combo"] == full_label]
            _draw_nsims_curves(ax, sub, f"fom_{key}", color, n_cosmo_values=n_cosmo_values)
        ax.set_title(FOB_LABELS[key], fontsize=18, fontweight="bold", pad=8)
        ax.set_ylabel("FoM", fontsize=14)
        ax.set_yscale("log")
        ax.set_xscale("log")
    for ax, key in zip(axes[1], FOB_KEYS):
        for full_label, color in zip(full_labels, COLORS_FID):
            sub = df_all[df_all["stat_combo"] == full_label]
            _draw_nsims_curves(ax, sub, f"fob_{key}", color, n_cosmo_values=n_cosmo_values)
        ax.axhline(gauss_ref, color="k", ls="--", lw=1, alpha=0.7)
        ax.axhline(lim1, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_ylabel("FoB", fontsize=14)
        ax.set_xscale("log")
    fob_vals = []
    for key in FOB_KEYS:
        col = f"fob_{key}"
        if col in df_all.columns:
            fob_vals.extend(df_all[col].dropna().tolist())
    if fob_vals:
        ymin, ymax = float(np.nanmin(fob_vals)), float(np.nanmax(fob_vals))
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
        fob_ylim = (max(0.0, ymin - pad), ymax + pad)
        for ax in axes[1]:
            ax.set_ylim(fob_ylim)
    for ax in axes[1]:
        ax.set_xlabel(XLABEL_NSIMS, fontsize=14)
    for ax in axes.ravel():
        ax.tick_params(labelsize=12)
    stat_handles = [
        plt.Line2D([0], [0], color=c, ls="-", lw=1.5)
        for c in COLORS_FID
    ]
    fig.tight_layout()
    _conv_legend_outside(
        fig, stat_handles=stat_handles, stat_labels=short_labels,
        n_cosmo_values=n_cosmo_values,
        fontsize=13, bbox_to_anchor=(0.91, 0.5), right=0.90,
    )
    if save_name is not None:
        save_figure(fig, save_name)
    plt.show()


def plot_conv_mse_delta_nsims(df_all, *, save_name=None):
    """Per-param median MSE (top) and median (pred−true) (bottom) vs N_train."""
    if df_all.empty:
        print("No data for MSE / delta convergence plot")
        return
    missing = [
        f"{p}_{k}" for k in PARAMS_TRACK for p in ("mse", "delta")
        if f"{p}_{k}" not in df_all.columns
    ]
    if missing:
        raise KeyError(
            "df_all missing columns " + ", ".join(missing)
            + " (stale cache? rebuild with new cache name / OVERWRITE_CACHE=True)"
        )
    full_labels = [utils_plot.get_stat_label(s) for s in STATISTICS_ARR_FID]
    short_labels = [get_stat_label_short(s) for s in STATISTICS_ARR_FID]
    n_cosmo_values = sorted(df_all["n_cosmo"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, key in zip(axes[0], PARAMS_TRACK):
        for full_label, color in zip(full_labels, COLORS_FID):
            sub = df_all[df_all["stat_combo"] == full_label]
            _draw_nsims_curves(ax, sub, f"mse_{key}", color, n_cosmo_values=n_cosmo_values)
        ax.set_title(FOB_LABELS[key], fontsize=16, fontweight="bold", pad=8)
        ax.set_ylabel(r"median MSE")
        ax.set_yscale("log")
        ax.set_xscale("log")
    for ax, key in zip(axes[1], PARAMS_TRACK):
        for full_label, color in zip(full_labels, COLORS_FID):
            sub = df_all[df_all["stat_combo"] == full_label]
            _draw_nsims_curves(ax, sub, f"delta_{key}", color, n_cosmo_values=n_cosmo_values)
        ax.axhline(0.0, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_ylabel(r"median $(\hat{\theta} - \theta_{\mathrm{true}})$")
        ax.set_xscale("log")
    for ax in axes[1]:
        ax.set_xlabel(XLABEL_NSIMS)
    stat_handles = [
        plt.Line2D([0], [0], color=c, ls="-", lw=1.5)
        for c in COLORS_FID
    ]
    fig.tight_layout()
    _conv_legend_outside(
        fig, stat_handles=stat_handles, stat_labels=short_labels,
        n_cosmo_values=n_cosmo_values,
        fontsize=11, bbox_to_anchor=(0.91, 0.5), right=0.90,
    )
    if save_name is not None:
        save_figure(fig, save_name)
    plt.show()


def _place_full_contour_legend(fig, *, fontsize=16):
    """Put the legend at the top-right of the triangle (figure coordinates)."""
    ax_leg = next((ax for ax in fig.axes if ax.get_legend() is not None), None)
    if ax_leg is None:
        return
    old = ax_leg.get_legend()
    handles = old.legend_handles
    labels = [t.get_text() for t in old.get_texts()]
    old.remove()
    fig.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        bbox_transform=fig.transFigure,
        fontsize=fontsize,
        frameon=False,
        labelcolor="black",
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.0,
        labelspacing=0.35,
    )


def plot_all_params(
    tags_inf, labels, colors, tags_test, keep, theta_obs, param_vary, *,
    truth_exclude=None, truth_color="k", save_name=None,
    fontsize_legend=16,
    legend_location=None,
    legend_loc=None,
    loc_legend=None,
):
    idxs = [i for i, k in enumerate(keep) if k]
    if not idxs:
        print("No samples for full-posterior plot")
        return
    extents_all = {
        **genp.get_bounds("cosmo"), **genp.get_bounds("bias"),
        **genp.get_bounds("Anoise", anoise_option=ANOISE_OPTION),
    }
    extents_all["b1"] = extents_contours["b1"]
    theta_vec = np.asarray(theta_obs, dtype=float).reshape(-1)
    if truth_exclude:
        exclude = set(truth_exclude)
        truth_loc = {
            pn: float(theta_vec[param_vary.index(pn)])
            for pn in param_vary
            if pn not in exclude and np.isfinite(theta_vec[param_vary.index(pn)])
        }
        truth_kwargs = dict(
            add_truth=True, truth_locations=[truth_loc], theta_obs_true=None,
            truth_colors=[truth_color],
        )
    else:
        truth_kwargs = dict(add_truth=True, theta_obs_true=theta_vec)
    fig = plotter.plot_contours_inf(
        param_names=list(param_vary), idx_obs=idx_obs,
        inf_methods=["sbi"] * len(idxs),
        tags_inf=[tags_inf[i] for i in idxs], tags_test=[tags_test[i] for i in idxs],
        colors=[colors[i] for i in idxs], labels=[labels[i] for i in idxs],
        title=None, extents=extents_all, figsize=FIGSIZE_ALL_PARAMS,
        fontsize_legend=fontsize_legend,
        legend_location=legend_location,
        legend_loc=legend_loc,
        loc_legend=loc_legend,
        unreparameterize=True,
        show=False,
        **truth_kwargs,
    )
    if fig is not None and legend_location is None and loc_legend is None:
        _place_full_contour_legend(fig, fontsize=fontsize_legend)
    if save_name is not None and fig is not None:
        save_figure(fig, save_name)
    plt.show()


# --- Fig 5 PP helpers ---

def _param_latex_body(pn):
    s = utils_plot.param_label_dict[pn]
    return s[1:-1] if s.startswith("$") and s.endswith("$") else s


def _physical_from_rp(arr):
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    out[..., 1] = arr[..., 1]
    out[..., 2] = arr[..., 2] / arr[..., 1]
    return out


def _ranks(samples_do, truth_o):
    return np.mean(samples_do < truth_o[np.newaxis, :], axis=0)


def _emp_coverage(ranks, cred_levels):
    a_min = 2.0 * np.abs(ranks[np.isfinite(ranks)] - 0.5)
    a_min.sort()
    return np.searchsorted(a_min, cred_levels, side="right") / a_min.size


def _gaussian_pp_miscal(cred, k):
    """PP curve if truth is Gaussian but reported σ = k × true σ."""
    from scipy.special import erfinv, erf
    cred = np.asarray(cred, dtype=float)
    return erf(float(k) * erfinv(cred))


def _gaussian_ref_pairs(sigma_scale_refs):
    """Group k-factors into (k_lo, k_hi) pairs for legend/styling."""
    refs = list(sigma_scale_refs)
    if not refs:
        return []
    if isinstance(refs[0], (tuple, list)) and len(refs[0]) == 2:
        pairs = [(float(a), float(b)) for a, b in refs]
    else:
        ks = sorted(float(k) for k in refs)
        los = [k for k in ks if k < 1.0]
        his = [k for k in ks if k > 1.0]
        his_desc = list(reversed(his))
        n = min(len(los), len(his_desc))
        pairs = list(zip(los[:n], his_desc[:n]))
        extras = los[n:] + his_desc[n:] + [k for k in ks if k == 1.0]
        for k in extras:
            pairs.append((k, k))
    pairs.sort(key=lambda p: max(abs(np.log(p[0])), abs(np.log(p[1]))))
    return pairs


def _load_key_samples(tag_inf, tag_test, names_need_rp):
    """mmap file and materialize only needed columns → ~3/11 of I/O."""
    dir_run = dir_sbi / f"sbi{tag_inf}"
    fn = dir_run / f"samples_test{tag_test}_pred.npy"
    names_samp = list(np.loadtxt(dir_run / "param_names.txt", dtype=str))
    idxs = [names_samp.index(pn) for pn in names_need_rp]
    mm = np.load(fn, mmap_mode="r")
    if mm.ndim == 2:
        sub = np.asarray(mm[:, idxs], dtype=np.float64)[:, None, :]
    else:
        sub = np.asarray(mm[:, :, idxs], dtype=np.float64)
    return sub


def _load_key_samples_ensemble(statistics, mask, names_need_rp):
    """K=3 coverage mixture; materialize only needed columns."""
    samples, names = _load_ensemble_coverage_for_combo(statistics, mask)
    names = list(names)
    idxs = [names.index(pn) for pn in names_need_rp]
    return np.asarray(samples[:, :, idxs], dtype=np.float64)


def _add_pp_split_legends(ax, stat_labels):
    """Stats upper-left; Gaussian miscalibration refs lower-right (first panel only)."""
    stat_set = set(stat_labels)
    h_all, l_all = ax.get_legend_handles_labels()
    h_stat, l_stat, h_gauss, l_gauss = [], [], [], []
    for h, lab in zip(h_all, l_all):
        if not lab:
            continue
        if lab.startswith("Gaussian"):
            h_gauss.append(h)
            l_gauss.append(lab)
        elif lab in stat_set:
            h_stat.append(h)
            l_stat.append(lab)
    leg_stat = ax.legend(
        h_stat, l_stat, loc="upper left", fontsize=10,
        frameon=False, labelcolor="black",
    )
    ax.add_artist(leg_stat)
    ax.legend(
        h_gauss, l_gauss, loc="lower right", fontsize=8,
        frameon=False, labelcolor="black",
    )


def _plot_pp_overlay(
    pp_emp, save_name, *, param_names_plot, colors, labels, cred_levels,
    title_suffix=None, sigma_scale_refs=PP_GAUSSIAN_SIGMA_REFS,
):
    n_r = pp_emp.shape[0]
    n_cols = len(param_names_plot)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(4.6 * n_cols, 3.3),
        gridspec_kw={"wspace": 0.05},
    )
    axes = np.atleast_1d(axes)
    pairs = _gaussian_ref_pairs(sigma_scale_refs) if sigma_scale_refs is not None else []
    if pairs:
        # Wider grey span so nearby miscalibration refs read more distinctly.
        greys = np.linspace(0.05, 0.72, len(pairs))
    for c in range(n_cols):
        ax = axes[c]
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, zorder=1)
        for i_p, (k_lo, k_hi) in enumerate(pairs):
            grey = str(greys[i_p])
            lab = None
            if c == 0:
                if np.isclose(k_lo, k_hi):
                    lab = rf"Gaussian ${k_lo:g}\sigma$"
                else:
                    lab = rf"Gaussian ${k_lo:g}\sigma/{k_hi:g}\sigma$"
            for k in dict.fromkeys((k_lo, k_hi)):
                ax.plot(
                    cred_levels, _gaussian_pp_miscal(cred_levels, k),
                    color=grey, lw=1.15, ls=":",
                    label=lab,
                    zorder=1.5,
                )
                lab = None
        for r in range(n_r):
            ax.plot(
                cred_levels, pp_emp[r, c],
                color=colors[r], lw=1.5,
                label=labels[r] if c == 0 else None,
                zorder=2,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ticks = np.arange(0, 1.01, 0.2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("credibility level", fontsize=12)
        ax.set_ylabel("empirical coverage", fontsize=12, labelpad=2.5)
        if c == 0:
            _add_pp_split_legends(ax, labels)
        title = rf"{utils_plot.param_label_dict[param_names_plot[c]]}"
        if title_suffix:
            title = rf"{title} {title_suffix}"
        ax.set_title(title, fontsize=15, fontweight="bold")
    plt.tight_layout(w_pad=1.0)
    save_figure(fig, save_name)
    plt.show()


def _coverage_keep_lists():
    """Shared fig4/fig5 filtering — require K=3 coverage ensemble files per combo."""
    tags_inf, labels, colors, tags_test, _keep_top, cosmo_vary, bias_vary, param_vary = (
        setup_coverage_test()
    )
    keep = []
    for statistics, mask in zip(STATISTICS_ARR_FID, TAGS_MASK_FID):
        keep.append(ensemble_coverage_members_exist(statistics, mask))
    tags_test_ok = [tags_test[i] for i, k in enumerate(keep) if k]
    tags_inf_ok = [tags_inf[i] for i, k in enumerate(keep) if k]
    colors_ok = [colors[i] for i, k in enumerate(keep) if k]
    stats_ok = [STATISTICS_ARR_FID[i] for i, k in enumerate(keep) if k]
    masks_ok = [TAGS_MASK_FID[i] for i, k in enumerate(keep) if k]
    labels_abbrev = [get_stat_label_short(s) for s in stats_ok]
    print(f"coverage ensemble keep (K={N_ENSEMBLE_K}):", keep)
    return {
        "tags_inf": tags_inf,
        "labels": labels,
        "colors": colors,
        "tags_test": tags_test,
        "keep": keep,
        "cosmo_vary": cosmo_vary,
        "bias_vary": bias_vary,
        "param_vary": param_vary,
        "tags_test_ok": tags_test_ok,
        "tags_inf_ok": tags_inf_ok,
        "colors_ok": colors_ok,
        "stats_ok": stats_ok,
        "masks_ok": masks_ok,
        "labels_abbrev": labels_abbrev,
    }


def _load_ensemble_coverage_for_combo(statistics, mask):
    """K=3 concatenated coverage samples + param names for one fiducial combo."""
    samples, missing = load_ensemble_member_samples(
        statistics, mask, test_mode="coverage", k_members=N_ENSEMBLE_K,
    )
    if samples is None:
        raise FileNotFoundError(
            f"ensemble coverage missing for {get_stat_label_short(statistics)}: "
            + "; ".join(missing)
        )
    fn_pn = model_dir(statistics, mask, 0) / "param_names.txt"
    names = list(np.loadtxt(fn_pn, dtype=str))
    return samples, names


# =============================================================================
# Fig 2 — Statistics: training-set ranges
# =============================================================================

def plot_fig2_statistics_ranges(save_name="fig2_statistics_ranges"):
    def _build_fig2_arrays():
        """Load training bands + SHAMe overlays (slow I/O); cache the plot arrays."""
        shame_stats = {}
        for tag_mock in NBAR_TAGS_STATS:
            k_pk_s, y_pk_s, _ = data_loader.load_data_shame("pk", tag_mock)
            k_pgm_s, y_pgm_s, _ = data_loader.load_data_shame("pgm", tag_mock)
            _, y_b_s, _ = data_loader.load_data_shame("bispec", tag_mock)
            shame_stats[tag_mock] = {
                "pk": (np.asarray(k_pk_s), np.asarray(y_pk_s)),
                "pgm": (np.asarray(k_pgm_s), np.asarray(y_pgm_s)),
                "bispec": (np.arange(len(y_b_s)), np.asarray(y_b_s)),
            }
        train = {}
        for statistic in ("pk", "pgm", "bispec"):
            k, ys, _, _ = data_loader.load_data_muchisimocks(
                statistic, TAG_PARAMS_TRAIN, TAG_BIASPARAMS_TRAIN,
                tag_noise=TAG_NOISE_TRAIN, bx=bx, n_cosmo_max=n_cosmo_max_stats,
            )
            median, p1, p16, p84, p99 = stats_summary(ys)
            x = np.arange(len(median)) if statistic == "bispec" else np.asarray(k)
            train[statistic] = dict(
                x=x, median=median, p1=p1, p16=p16, p84=p84, p99=p99,
            )
        return {"train": train, "shame": shame_stats}

    cache_name = (
        f"fig2_stat_ranges_{TAG_PARAMS_TRAIN}{TAG_BIASPARAMS_TRAIN}"
        f"{TAG_NOISE_TRAIN}_bx{bx}_n{n_cosmo_max_stats}"
    )
    arrays = load_or_build(cache_name, _build_fig2_arrays)
    train, shame_stats = arrays["train"], arrays["shame"]

    def plot_shame_nbars(ax, statistic, *, with_labels=False):
        for tag_mock in NBAR_TAGS_STATS:
            x, y = shame_stats[tag_mock][statistic]
            ax.plot(
                x, y,
                color=NBAR_COLORS_STATS[tag_mock],
                linestyle=SHAME_LINESTYLE,
                linewidth=SHAME_LINEWIDTH,
                label=NBAR_LABELS_STATS[tag_mock] if with_labels else None,
                zorder=5 if tag_mock == TAG_MOCK_SHAME else 4,
            )

    def add_split_legends(ax):
        """Training percentiles upper-right; SHAMe n̄ lower-left (left panel only)."""
        h_all, l_all = ax.get_legend_handles_labels()
        train_keys = {"training set median", "16–84%", "1–99%"}
        h_train, l_train, h_shame, l_shame = [], [], [], []
        for h, lab in zip(h_all, l_all):
            if lab in train_keys:
                h_train.append(h)
                l_train.append(lab)
            else:
                h_shame.append(h)
                l_shame.append(lab)
        leg_train = ax.legend(
            h_train, l_train, loc="upper right", fontsize=LEGENDSIZE,
            frameon=False, labelcolor="black",
        )
        ax.add_artist(leg_train)
        ax.legend(
            h_shame, l_shame, loc="lower left",
            fontsize=LEGENDSIZE - 1,
            title="SHAMe mock",
            title_fontsize=LEGENDSIZE - 1,
            frameon=False, labelcolor="black",
        )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # P(k) — legends only on this panel
    t = train["pk"]
    plot_stat_band(
        axes[0], t["x"], t["median"], t["p1"], t["p16"], t["p84"], t["p99"],
        COLOR_STAT["pk"], with_labels=True,
    )
    plot_shame_nbars(axes[0], "pk", with_labels=True)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$k\,[h/\mathrm{Mpc}]$", fontsize=LABELSIZE)
    axes[0].set_ylabel(r"$P(k)$", fontsize=LABELSIZE)
    add_split_legends(axes[0])

    # P_gm(k)
    t = train["pgm"]
    plot_stat_band(
        axes[1], t["x"], t["median"], t["p1"], t["p16"], t["p84"], t["p99"],
        COLOR_STAT["pgm"], with_labels=False,
    )
    plot_shame_nbars(axes[1], "pgm", with_labels=False)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$k\,[h/\mathrm{Mpc}]$", fontsize=LABELSIZE)
    axes[1].set_ylabel(r"$P_{gm}(k)$", fontsize=LABELSIZE)

    # Bispectrum (triangle bin index; matches 2026-03-10_load_statistics)
    t = train["bispec"]
    plot_stat_band(
        axes[2], t["x"], t["median"], t["p1"], t["p16"], t["p84"], t["p99"],
        COLOR_STAT["bispec"], with_labels=False,
    )
    plot_shame_nbars(axes[2], "bispec", with_labels=False)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("triangle bin index", fontsize=LABELSIZE)
    axes[2].set_ylabel(r"$B(k_1,k_2,k_3)$", fontsize=LABELSIZE)

    for ax in axes:
        ax.tick_params(labelsize=TICKSIZE)

    plt.tight_layout()
    save_figure(fig, save_name)
    plt.show()


# =============================================================================
# Fig 3 — Cosmic variance contours
# =============================================================================

def plot_fig3_cosmic_variance_contours(save_name="fig3_cosmic_variance_contours"):
    tags_inf, labels, colors, tags_test, keep, _, theta_show, _ = setup_fixed_mean_test()
    print("keep (top-model samples on disk):", keep)
    print("ensemble keep (cvmean):", ensemble_samples_exist(test_mode="cvmean"))
    # Slightly above default LOC_LEGEND y so the legend sits a touch higher.
    loc_legend_fig3 = (LOC_LEGEND[0], LOC_LEGEND[1] + 0.08)
    plot_fiducial_ensemble_contours(
        test_mode="cvmean",
        theta_show=theta_show,
        extents=extents_contours,
        unreparameterize=True,
        save_name=save_name,
        title=None,
        loc_legend=loc_legend_fig3,
    )


def plot_fig3_cosmic_variance_contours_sigma8xb1(
    save_name="fig3_cosmic_variance_contours_sigma8xb1",
):
    tags_inf, labels, colors, tags_test, keep, _, theta_show, _ = setup_fixed_mean_test()
    i_s8 = PARAM_NAMES_ZOOM.index("sigma8_cold")
    i_b1 = PARAM_NAMES_ZOOM.index("b1")
    theta_show_rp = np.array(
        [
            theta_show[PARAM_NAMES_ZOOM.index("omega_cold")],
            theta_show[i_s8],
            theta_show[i_s8] * theta_show[i_b1],
        ],
        dtype=float,
    )
    extents_rp = {k: v for k, v in extents_contours.items() if k != "b1"}
    extents_rp["sigma8_cold_x_b1"] = [
        extents_contours["b1"][0] * theta_show[i_s8],
        extents_contours["b1"][1] * theta_show[i_s8],
    ]

    samples_list, labels_e, colors_e, tags_e, _ = load_fiducial_ensemble_samples_list(test_mode="cvmean")
    loc_legend_fig3 = (LOC_LEGEND[0], LOC_LEGEND[1] + 0.08)
    plot_fiducial_contours(
        tags_e, labels_e, colors_e, [""] * len(tags_e), [True] * len(tags_e),
        theta_show=theta_show_rp,
        extents=extents_rp,
        param_names=list(param_names_key_rp),
        unreparameterize=False,
        samples_list=samples_list,
        save_name=save_name,
        title=None,
        loc_legend=loc_legend_fig3,
    )


# =============================================================================
# Fig 4 — Coverage prediction bias
# =============================================================================

def plot_fig4_coverage_binned_diff(save_name="fig4_coverage_binned_diff"):
    param_names_plot = param_names_key
    cov = _coverage_keep_lists()
    colors_ok = cov["colors_ok"]
    stats_ok = cov["stats_ok"]
    masks_ok = cov["masks_ok"]
    labels_abbrev = cov["labels_abbrev"]
    cosmo_vary = cov["cosmo_vary"]
    bias_vary = cov["bias_vary"]
    param_vary = cov["param_vary"]

    def _build_fig4_coverage_arrays():
        names_show = list(param_names_key_rp)
        theta_pred_list, theta_true_list, covs_list = [], [], []
        names_u = None
        for statistics, mask in zip(stats_ok, masks_ok):
            samples_arr, param_names = _load_ensemble_coverage_for_combo(statistics, mask)
            param_names = list(param_names)
            i_show = [param_names.index(pn) for pn in names_show]
            samples_show = samples_arr[:, :, i_show]
            theta_pred, covs_pred, _, rows = _moments_from_samples_3d(
                samples_show, names_show,
            )
            theta_true_all = data_loader.load_theta_test(
                TAG_PARAMS_TEST_COV, TAG_BIASPARAMS_TEST_COV,
                cosmo_param_names_vary=cosmo_vary, bias_param_names_vary=bias_vary,
            )
            if theta_true_all.ndim == 1:
                theta_true_all = np.tile(theta_true_all, (samples_arr.shape[1], 1))
            theta_true_phys = theta_true_all[rows]
            if tag_reparam:
                theta_true_rp, names_rp = utils_inference.reparameterize_theta(
                    theta_true_phys, list(param_vary),
                )
                names_rp = list(names_rp)
                theta_true = np.column_stack(
                    [theta_true_rp[:, names_rp.index(pn)] for pn in names_show]
                )
            else:
                theta_true = np.column_stack(
                    [theta_true_phys[:, list(param_vary).index(pn)] for pn in names_show]
                )
            tp, tt, covs, names_u = utils_inference.unreparameterize_prediction_block(
                theta_pred, theta_true, names_show, covs_pred,
            )
            theta_pred_list.append(tp)
            theta_true_list.append(tt)
            covs_list.append(covs)
        return {
            "theta_pred_u": theta_pred_list,
            "theta_true_u": theta_true_list,
            "covs_u": covs_list,
            "names_u": list(names_u),
        }

    _fig4 = load_or_build("fig4_coverage_arrays_kp035_ensK3", _build_fig4_coverage_arrays)
    theta_pred_u = _fig4["theta_pred_u"]
    theta_true_u = _fig4["theta_true_u"]
    names_u = _fig4["names_u"]

    idxs_plot = [names_u.index(pn) for pn in param_names_plot]
    n_rows, n_cols = len(stats_ok), len(param_names_plot)

    fig, axes = plt.subplots(
        1, n_cols, figsize=(4.6 * n_cols, 3.3),
        gridspec_kw={"wspace": 0.35},
    )
    axes = np.atleast_1d(axes)

    for c, pp in enumerate(idxs_plot):
        ax = axes[c]
        body = _param_latex_body(param_names_plot[c])

        true_all = np.concatenate([theta_true_u[r][:, pp] for r in range(n_rows)])
        true_all = true_all[np.isfinite(true_all)]
        edges = np.linspace(np.min(true_all), np.max(true_all), n_bins_fig4 + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        half = 0.0
        ax.axhline(0.0, color="k", ls="--", lw=1.2, alpha=0.7, zorder=1)
        for r in range(n_rows):
            true_vals = theta_true_u[r][:, pp]
            y_vals = theta_pred_u[r][:, pp] - true_vals
            ok = np.isfinite(true_vals) & np.isfinite(y_vals)
            x, y = true_vals[ok], y_vals[ok]
            medians = np.full(n_bins_fig4, np.nan)
            p16s = np.full(n_bins_fig4, np.nan)
            p84s = np.full(n_bins_fig4, np.nan)
            for i in range(n_bins_fig4):
                m = (x >= edges[i]) & (x < edges[i + 1] if i < n_bins_fig4 - 1 else x <= edges[i + 1])
                if np.count_nonzero(m) == 0:
                    continue
                p16, med, p84 = np.percentile(y[m], [16, 50, 84])
                medians[i], p16s[i], p84s[i] = med, p16, p84
            finite = np.isfinite(medians)
            if np.any(finite):
                half = max(
                    half,
                    float(np.nanmax(np.abs(p16s[finite]))),
                    float(np.nanmax(np.abs(p84s[finite]))),
                )
                ax.fill_between(
                    centers[finite], p16s[finite], p84s[finite],
                    color=colors_ok[r], alpha=0.22, linewidth=0, zorder=2,
                )
                ax.plot(
                    centers[finite], medians[finite],
                    color=colors_ok[r], lw=1.5,
                    label=labels_abbrev[r] if c == 0 else None,
                    zorder=3,
                )

        ax.set_ylim(-1.05 * max(half, 1e-12), 1.05 * max(half, 1e-12))
        ax.set_xlabel(rf"${body}^{{\mathrm{{true}}}}$", fontsize=14)
        ax.set_ylabel(
            rf"${body}^{{\mathrm{{pred}}}} - {body}^{{\mathrm{{true}}}}$",
            fontsize=14, labelpad=1.5,
        )
        if c == 0:
            ax.legend(
                fontsize=11, loc="lower left", bbox_to_anchor=(0.0, -0.04),
                frameon=False, labelcolor="black",
            )

    plt.tight_layout(w_pad=3.0)
    save_figure(fig, save_name)
    plt.show()


# =============================================================================
# Fig 5 — Coverage PP plots
# =============================================================================

def plot_fig5_coverage_pp(save_name="fig5_coverage_pp"):
    param_names_plot = list(param_names_key)
    cov = _coverage_keep_lists()
    colors_ok = cov["colors_ok"]
    stats_ok = cov["stats_ok"]
    masks_ok = cov["masks_ok"]
    labels_abbrev = cov["labels_abbrev"]
    cosmo_vary = cov["cosmo_vary"]
    bias_vary = cov["bias_vary"]
    param_vary = cov["param_vary"]
    n_rows = len(stats_ok)
    n_cols = len(param_names_plot)
    cred_levels = np.linspace(0.0, 1.0, 51)

    theta_test_all = data_loader.load_theta_test(
        TAG_PARAMS_TEST_COV,
        TAG_BIASPARAMS_TEST_COV,
        cosmo_param_names_vary=cosmo_vary,
        bias_param_names_vary=bias_vary,
    )
    if tag_reparam:
        theta_test_all, names_theta = utils_inference.reparameterize_theta(
            theta_test_all, list(param_vary),
        )
    else:
        names_theta = list(param_vary)
    names_theta = list(names_theta)
    names_need_rp = ["omega_cold", "sigma8_cold", "sigma8_cold_x_b1"]
    i_t = [names_theta.index(pn) for pn in names_need_rp]
    theta_phys_all = _physical_from_rp(theta_test_all[:, i_t])

    def _build_fig5_pp_emp():
        pp = np.full((n_rows, n_cols, cred_levels.size), np.nan)
        for r in range(n_rows):
            samples_rp = _load_key_samples_ensemble(stats_ok[r], masks_ok[r], names_need_rp)
            ok = np.all(np.isfinite(samples_rp.mean(axis=0)), axis=1)
            samples_phys = _physical_from_rp(samples_rp[:, ok, :])
            theta_phys = theta_phys_all[ok]
            for c in range(n_cols):
                ranks = _ranks(samples_phys[:, :, c], theta_phys[:, c])
                pp[r, c] = _emp_coverage(ranks, cred_levels)
        return pp

    pp_emp_arr = load_or_build("fig5_pp_emp_arr_kp035_ensK3", _build_fig5_pp_emp)
    _plot_pp_overlay(
        pp_emp_arr, save_name,
        param_names_plot=param_names_plot,
        colors=colors_ok, labels=labels_abbrev, cred_levels=cred_levels,
    )


def plot_fig5_coverage_pp_center(save_name=None, n_center_pts=None):
    if n_center_pts is None:
        n_center_pts = n_center
    if save_name is None:
        save_name = f"fig5_coverage_pp_center{n_center_pts}"

    param_names_plot = list(param_names_key)
    cov = _coverage_keep_lists()
    colors_ok = cov["colors_ok"]
    stats_ok = cov["stats_ok"]
    masks_ok = cov["masks_ok"]
    labels_abbrev = cov["labels_abbrev"]
    cosmo_vary = cov["cosmo_vary"]
    bias_vary = cov["bias_vary"]
    param_vary = cov["param_vary"]
    n_rows = len(stats_ok)
    n_cols = len(param_names_plot)
    cred_levels = np.linspace(0.0, 1.0, 51)

    theta_test_all = data_loader.load_theta_test(
        TAG_PARAMS_TEST_COV,
        TAG_BIASPARAMS_TEST_COV,
        cosmo_param_names_vary=cosmo_vary,
        bias_param_names_vary=bias_vary,
    )
    if tag_reparam:
        theta_test_all, names_theta = utils_inference.reparameterize_theta(
            theta_test_all, list(param_vary),
        )
    else:
        names_theta = list(param_vary)
    names_theta = list(names_theta)
    names_need_rp = ["omega_cold", "sigma8_cold", "sigma8_cold_x_b1"]
    i_t = [names_theta.index(pn) for pn in names_need_rp]
    theta_phys_all = _physical_from_rp(theta_test_all[:, i_t])

    _bounds_key = [extents[pn] for pn in param_names_plot]
    _center = np.array([0.5 * (lo + hi) for lo, hi in _bounds_key])
    _scale = np.array([0.5 * (hi - lo) for lo, hi in _bounds_key])
    _dist3d = np.sqrt(np.sum(((theta_phys_all - _center) / _scale) ** 2, axis=1))
    idxs_c = np.argsort(_dist3d)[:n_center_pts]
    mask_center = np.zeros(theta_phys_all.shape[0], dtype=bool)
    mask_center[idxs_c] = True
    print(
        f"Center-{n_center_pts}: prior midpoints {_center}, "
        f"max normalized 3D dist = {_dist3d[idxs_c].max():.3f}"
    )

    def _build_fig5_pp_emp_center():
        pp = np.full((n_rows, n_cols, cred_levels.size), np.nan)
        for r in range(n_rows):
            samples_rp = _load_key_samples_ensemble(stats_ok[r], masks_ok[r], names_need_rp)
            ok = np.all(np.isfinite(samples_rp.mean(axis=0)), axis=1) & mask_center
            print(f"  {labels_abbrev[r]}: {np.count_nonzero(ok)}/{n_center_pts} center points usable")
            samples_phys = _physical_from_rp(samples_rp[:, ok, :])
            theta_phys = theta_phys_all[ok]
            for c in range(n_cols):
                ranks = _ranks(samples_phys[:, :, c], theta_phys[:, c])
                pp[r, c] = _emp_coverage(ranks, cred_levels)
        return pp

    pp_emp_center = load_or_build(
        f"fig5_pp_emp_arr_kp035_ensK3_center{n_center_pts}", _build_fig5_pp_emp_center,
    )
    _plot_pp_overlay(
        pp_emp_center, save_name,
        param_names_plot=param_names_plot,
        colors=colors_ok, labels=labels_abbrev, cred_levels=cred_levels,
    )


# =============================================================================
# Fig 6 — SHAMe OOD contours
# =============================================================================

def plot_fig6_shame_contours(save_name="fig6_shame_contours"):
    tags_inf, labels, colors, tags_test, keep, param_vary, theta_show = setup_shame_test()
    print("keep (top-model samples on disk):", keep)
    print("ensemble keep (shame):", ensemble_samples_exist(test_mode="shame", tag_mock=TAG_MOCK_SHAME))
    loc_legend_fig6 = (LOC_LEGEND[0], LOC_LEGEND[1] + 0.08)
    plot_fiducial_ensemble_contours(
        test_mode="shame",
        tag_mock=TAG_MOCK_SHAME,
        theta_show=theta_show,
        extents=extents_contours,
        save_name=save_name,
        title=None,
        loc_legend=loc_legend_fig6,
    )


# =============================================================================
# Fig 7 — Training noise and number density
# =============================================================================

def plot_fig7_noise_impact_contours_nbars(save_name="fig7_noise_impact_contours_nbars"):
    """Noisy OOD SHAMe × n̄ + noisy/noiseless CV-mean overlays; noisy uses K=3 ensemble."""
    statistics_full = list(STATISTICS_FULL_ROW)
    tag_mask_full = TAG_MASK_FULL
    nbar_tags = NBAR_TAGS_SCALE
    shades_noisy = _nbar_shades(COLOR_NOISY)
    color_nbar_fid = shades_noisy[nbar_tags.index(TAG_MOCK_SHAME)]

    samples_list, colors_b, labels_b, shades_b = [], [], [], []
    linestyles_b, linewidths_b = [], []
    missing_b, existing_b = [], []
    # Dummy tags for plot_contours_inf labeling (samples_list path).
    tags_inf_b, tags_test_b = [], []

    # Noisy OOD SHAMe × 3 nbars (K=3 ensemble mixtures).
    for i, tag_mock in enumerate(nbar_tags):
        samples, missing = load_ensemble_member_samples(
            statistics_full, tag_mask_full,
            test_mode="shame", tag_mock=tag_mock, k_members=N_ENSEMBLE_K,
            n_total=N_ENSEMBLE_DRAWS,
        )
        label = f"noisy × {NBAR_LABELS[tag_mock]}"
        if samples is None:
            missing_b.append((label, "; ".join(missing)))
            continue
        existing_b.append(label)
        fn_pn = model_dir(statistics_full, tag_mask_full, 0) / "param_names.txt"
        names = list(np.loadtxt(fn_pn, dtype=str))
        arr = samples if samples.ndim == 2 else samples[:, 0, :]
        samples_list.append((arr, names))
        colors_b.append(shades_noisy[i])
        labels_b.append(f"SHAMe mock, {NBAR_LABELS[tag_mock]}")
        shades_b.append(True)
        linestyles_b.append("-")
        linewidths_b.append(1.0)
        tags_inf_b.append(f"_ensK{N_ENSEMBLE_K}_noisy_{tag_mock}")
        tags_test_b.append("")

    # Noisy fixed-cosmo mean (K=3 ensemble).
    samples_nm, missing_nm = load_ensemble_member_samples(
        statistics_full, tag_mask_full,
        test_mode="cvmean", k_members=N_ENSEMBLE_K, n_total=N_ENSEMBLE_DRAWS,
    )
    if samples_nm is not None:
        fn_pn = model_dir(statistics_full, tag_mask_full, 0) / "param_names.txt"
        names = list(np.loadtxt(fn_pn, dtype=str))
        arr = samples_nm if samples_nm.ndim == 2 else samples_nm[:, 0, :]
        samples_list.append((arr, names))
        colors_b.append(color_nbar_fid)
        labels_b.append("mean-CV, noisy (matched to $\\bar{n}=2.2\\times10^{-4}$ mock)")
        shades_b.append(False)
        linestyles_b.append(":")
        linewidths_b.append(1.0)
        tags_inf_b.append(f"_ensK{N_ENSEMBLE_K}_noisy_cvmean")
        tags_test_b.append("")
        existing_b.append("noisy × shame mean-CV")
    else:
        missing_b.append(("noisy × shame mean-CV", "; ".join(missing_nm)))

    # Noiseless fixed-cosmo mean — top-1 if present (KP035 noiseless not trained yet).
    tags_inf_noiseless, _, _, _ = utils_plot.setup_inference_tags(
        data_mode=data_mode, tag_params=TAG_PARAMS_TRAIN,
        tag_biasparams="_biasnest_p4_n320000", statistics_arr=[statistics_full],
        bx=bx, tag_noise=None, tag_reparam=tag_reparam, n_train=n_train,
        tags_mask=[tag_mask_full],
    )
    tag_inf_noiseless = tags_inf_noiseless[0] + TAG_INF_BEST_SUFFIX
    tag_test_noiseless_mean = utils_plot.setup_test_tags(
        data_mode=data_mode, tag_params_test=TAG_PARAMS_TEST_FIXED,
        tags_biasparams_test="_biasshame_p0_n1", tag_stats_arr=[TAG_STATS_FULL],
        tag_noise_test=None, tag_datagen_test=TAG_DATAGEN_TEST_MEAN,
        tags_mask_test=[tag_mask_full],
    )[0]
    fn_nl = sample_fn(tag_inf_noiseless, tag_test_noiseless_mean)
    has_noiseless = fn_nl.exists()
    if has_noiseless:
        arr = np.load(fn_nl)
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        names = list(np.loadtxt(dir_sbi / f"sbi{tag_inf_noiseless}" / "param_names.txt", dtype=str))
        samples_list.append((arr, names))
        colors_b.append(color_nbar_fid)
        labels_b.append("mean-CV, noiseless (matched to $\\bar{n}=2.2\\times10^{-4}$ mock)")
        shades_b.append(False)
        linestyles_b.append("--")
        linewidths_b.append(1.5)
        tags_inf_b.append(tag_inf_noiseless)
        tags_test_b.append("")
        existing_b.append("noiseless × shame mean")
    else:
        missing_b.append(("noiseless × shame mean", str(fn_nl)))

    if missing_b:
        print("Missing fig 7 sample files:")
        for label, fn in missing_b:
            print(f"  - {label}: {fn}")
    print(f"Found {len(existing_b)}/{len(existing_b) + len(missing_b)} (K={N_ENSEMBLE_K} ensemble for noisy).")

    n_noisy_nbar = sum(1 for s, lab in zip(shades_b, labels_b) if s and "SHAMe" in lab)
    if n_noisy_nbar == 0:
        print("ERROR: need noisy SHAMe nbar ensemble samples.")
        return
    if not has_noiseless:
        print("WARNING: noiseless CV-mean at kp0.35 missing — plotting noisy ensemble only.")

    cosmo_vary, bias_vary, param_vary = utils_plot.load_training_params(
        TAG_PARAMS_TRAIN, TAG_BIASPARAMS_TRAIN, bx=bx,
    )
    truth_locs, truth_cols = [], []
    for i, tag_mock in enumerate(nbar_tags):
        theta_ood = data_loader.load_theta_ood(
            DATA_MODE_TEST_SHAME, tag_mock,
            cosmo_param_names_vary=cosmo_vary,
            bias_param_names_vary=bias_vary,
        )
        theta_vec = theta_ood[idx_obs] if np.ndim(theta_ood) == 2 else theta_ood
        truth_locs.append({pn: float(theta_vec[param_vary.index(pn)]) for pn in PARAM_NAMES_ZOOM})
        truth_cols.append(shades_noisy[i])

    extents_b_nbar = extents_contours.copy()
    extents_b_nbar["b1"] = [0.3, 0.60]

    fig = plotter.plot_contours_inf(
        param_names=PARAM_NAMES_ZOOM, idx_obs=idx_obs, theta_obs_true=None,
        inf_methods=["sbi"] * len(samples_list),
        tags_inf=tags_inf_b, tags_test=tags_test_b,
        colors=colors_b, labels=labels_b, shades=shades_b,
        linestyles=linestyles_b,
        linewidths=linewidths_b,
        title=None,
        extents=extents_b_nbar,
        figsize=FIGSIZE_CORNER, unreparameterize=True,
        fontsize_legend=9,
        legend_location=(0, 1),
        legend_loc="upper left",
        loc_legend=None,
        samples_list=samples_list,
        add_truth=True,
        truth_locations=truth_locs,
        truth_colors=truth_cols,
        show=False,
    )
    if fig is not None:
        save_figure(fig, save_name)
    plt.show()


# =============================================================================
# Fig 8 — Scale dependence FoB (fixed kp=0.35)
# =============================================================================

def plot_fig8_scale_dependence_fob3_nbars(
    *,
    nbar_tags=None,
    save_name="fig8_scale_dependence_fob3_nbars",
    report_missing=True,
    param_names_fob=None,
    fob_keys=None,
    ylabel=None,
):
    """Fig8 scale FoB3 with multiple SHAMe number densities as color shades."""
    if nbar_tags is None:
        nbar_tags = NBAR_TAGS_SCALE
    ref3 = fob_chi2_sqrt_limit(3)
    load_shame_fob3._param_names_fob = param_names_fob
    load_shame_fob3._fob_keys = fob_keys
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharey=True)
    nbar_legend_shades = _nbar_shades("0.25")

    panel_specs = [
        (
            axes[0],
            _sweep_bispec_fob3,
            r"$k_\mathrm{max}^{B_\mathrm{ggg}}$",
            True,
        ),
        (
            axes[1],
            _sweep_pgm_fob3,
            r"$k_\mathrm{max}^{P_\mathrm{gm}}$",
            False,
        ),
    ]

    for ax, sweep_fn, xlabel, bispec_sweep in panel_specs:
        for statistics_row, color in STAT_COMBOS_SCALE:
            label = get_stat_label_short(statistics_row)
            color_shades = _nbar_shades(color)
            for j, tag_mock in enumerate(nbar_tags):
                c = color_shades[j]
                # Stat labels only on the fiducial nbar (mid shade) to avoid clutter.
                line_label = label if tag_mock == TAG_MOCK_SHAME else None
                const_val = _constant_fob3_on_sweep(
                    statistics_row,
                    bispec_sweep=bispec_sweep,
                    tag_mock=tag_mock,
                    report_missing=report_missing,
                )
                if const_val is not None:
                    ax.axhline(
                        const_val / ref3,
                        color=c,
                        lw=1.2,
                        ls="-",
                        alpha=0.95,
                        label=line_label,
                        zorder=1,
                    )
                    continue
                xs, ys = sweep_fn(
                    statistics_row, tag_mock=tag_mock, report_missing=report_missing,
                )
                if xs.size == 0:
                    print(
                        f"MISSING all FoB3 points | panel={xlabel} | stats={label} | "
                        f"nbar={tag_mock}"
                    )
                    continue
                order = np.argsort(xs)
                xs, ys = xs[order], ys[order] / ref3
                ax.plot(
                    xs, ys, color=c, ls="-", marker="o", lw=1.3, ms=5,
                    label=line_label, zorder=2 + j,
                )
        ax.set_xlabel(xlabel, fontsize=14)

    for ax in axes:
        ax.axvline(0.25, color="0.6", ls="--", lw=1.0, zorder=0)
        ax.set_xlim(*KMAX_XLIM)
        ax.tick_params(labelsize=12)
        ax.axhline(1.0, color="0.35", ls=":", lw=1.0, zorder=0)
    ylab = FOB3_YLABEL if ylabel is None else ylabel
    axes[0].set_ylabel(ylab, fontsize=14)
    axes[1].set_ylabel(ylab, fontsize=14)
    axes[0].set_ylim(0.0, 2.5)

    handles_stat_raw, labels_stat = axes[0].get_legend_handles_labels()
    handles_stat = [
        Line2D([0], [0], color=h.get_color(), ls="-", lw=2.0)
        for h in handles_stat_raw
    ]
    handles_nbar = [
        Line2D([0], [0], color=nbar_legend_shades[j], ls="-", lw=2.0)
        for j, _t in enumerate(nbar_tags)
    ]
    labels_nbar = [NBAR_LABELS[t] for t in nbar_tags]
    # Stats upper-left / n̄ upper-right on bottom panel; slightly smaller & higher.
    leg_stat = axes[1].legend(
        handles_stat, labels_stat,
        fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 1.02),
        labelcolor="black", frameon=False,
    )
    axes[1].add_artist(leg_stat)
    axes[1].legend(
        handles_nbar, labels_nbar,
        fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.02),
        labelcolor="black", frameon=False,
    )
    fig.tight_layout()
    if save_name is not None:
        save_figure(fig, save_name)
    plt.show()


# Keep notebook-era alias
plot_scale_fob3_shame_nbars = plot_fig8_scale_dependence_fob3_nbars


# =============================================================================
# Fig 9 — Overall k_max FoM for Ωc, σ8, b1 (3 panels)
# =============================================================================

def plot_fig9_scale_dependence_overall_kmax_fom3_nbars(
    *,
    test_mode="shame",
    nbar_tags=None,
    save_name="fig9_scale_dependence_overall_kmax_fom_marg_nbars_ensemble_shame",
    report_missing=False,
):
    """Ensemble overall k_max marginal FoM for Ωc, σ8, b1 — 3-panel (8-19 style)."""
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    nbar_tags = nbar_tags or NBAR_TAGS_SCALE

    for ax, param_key in zip(axes, FOB_KEYS):
        for statistics_row, color in STAT_COMBOS_OVERALL:
            label = get_stat_label_short(statistics_row)
            color_shades = _nbar_shades(color)
            for j, tag_mock_iter in enumerate(nbar_tags):
                c = color_shades[j]
                line_label = label if tag_mock_iter == TAG_MOCK_SHAME else None
                xs, ys = _overall_kmax_fom_marg_ensemble(
                    statistics_row, param_key, test_mode=test_mode,
                    tag_mock=tag_mock_iter, report_missing=report_missing,
                )
                if xs.size == 0:
                    continue
                order = np.argsort(xs)
                xs, ys = xs[order], ys[order]
                ax.plot(
                    xs, ys, color=c, ls="-", marker="o", lw=1.3, ms=5,
                    label=line_label, zorder=2 + j,
                )
        ax.axvline(kmax_plot_x_pk(0.25), color="0.6", ls="--", lw=1.0, zorder=0)
        ax.set_ylabel(FOM_MARG_LABELS[param_key], fontsize=14)
        ax.set_yscale("log")
        ax.set_xlim(*KMAX_XLIM)
        ax.tick_params(labelsize=12)

    axes[-1].set_xlabel(KMAX_OVERALL_XLABEL, fontsize=12)

    handles_stat_raw, labels_stat = axes[0].get_legend_handles_labels()
    handles_stat = [
        Line2D([0], [0], color=h.get_color(), ls="-", lw=2.0, marker="o", ms=5)
        for h in handles_stat_raw
    ]
    nbar_legend_shades = _nbar_shades("0.25")
    handles_nbar = [
        Line2D([0], [0], color=nbar_legend_shades[j], ls="-", lw=2.0, marker="o", ms=5)
        for j, _t in enumerate(nbar_tags)
    ]
    labels_nbar = [NBAR_LABELS[t] for t in nbar_tags]
    # Stats upper-left / n̄ lower-right on top panel (8-19 ensemble style).
    leg_stat = axes[0].legend(
        handles_stat, labels_stat,
        fontsize=9, loc="upper left",
        labelcolor="black", frameon=False,
    )
    axes[0].add_artist(leg_stat)
    axes[0].legend(
        handles_nbar, labels_nbar,
        fontsize=9, loc="lower right",
        labelcolor="black", frameon=False,
        title=r"$\bar n$", title_fontsize=9,
    )
    fig.tight_layout()
    if save_name is not None:
        save_figure(fig, save_name)
    plt.show()


# Keep notebook-era aliases
plot_scale_overall_kmax_fom_joint_ensemble = plot_fig9_scale_dependence_overall_kmax_fom3_nbars
plot_fig8b_scale_dependence_overall_kmax_fom3d_full_nbars = (
    plot_fig9_scale_dependence_overall_kmax_fom3_nbars
)


# =============================================================================
# Fig 10 — Convergence Pgg+Pgm+B (3D FoM/FoB)
# =============================================================================

def plot_fig10_convergence_pgg_b_pgm(save_name="fig10_convergence_pgg_b_pgm"):
    # Cache bumped: K=3 ensemble mix when all members exist.
    df_conv_full = load_or_build(
        "df_conv_pk_bispec_pgm_kp0.35_kb0.25_kpgm0.25_b1phys_err_ensK3",
        lambda: collect_coverage_grid(
            STATISTICS_FULL_CONV, TAG_MASKS_FULL_CONV, n_cosmo_arr, bx_arr,
        ),
    )
    print(f"$P_{{gg}}+P_{{gm}}+B_{{ggg}}$: {len(df_conv_full)} grid points loaded")
    if not df_conv_full.empty and "status" in df_conv_full.columns:
        print(df_conv_full["status"].value_counts().to_dict())
    plot_conv_fom3d_fob3_nsims(
        df_conv_full,
        COLORS_FID[3],
        save_name=save_name,
    )
    return df_conv_full


plot_fig9_convergence_pgg_b_pgm = plot_fig10_convergence_pgg_b_pgm


# =============================================================================
# Fig 11 — Convergence all stat combos
# =============================================================================

def plot_fig11_convergence_all_combos(save_name="fig11_convergence_all_combos"):
    frames = []
    for statistics, tag_masks, label in zip(
        STATISTICS_ARR_FID, TAGS_MASK_FID,
        [utils_plot.get_stat_label(s) for s in STATISTICS_ARR_FID],
    ):
        cache_name = (
            "df_conv_" + "_".join(statistics) + (tag_masks or "_nomask")
            + "_b1phys_err_ensK3"
        )
        df = load_or_build(
            cache_name,
            lambda s=statistics, m=tag_masks: collect_coverage_grid(s, m, n_cosmo_arr, bx_arr),
        )
        if df.empty:
            print(f"No coverage grid for {label}")
            continue
        df = df.copy()
        df["stat_combo"] = label
        frames.append(df)
        n_inc = int((df["status"] != "complete").sum()) if "status" in df.columns else 0
        print(f"{label}: {len(df)} grid points loaded ({n_inc} incomplete)")
    df_conv_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    plot_conv_all_combos_nsims(df_conv_all, save_name=save_name)
    return df_conv_all


plot_fig10_convergence_all_combos = plot_fig11_convergence_all_combos


# =============================================================================
# Appendix A — mean vs indiv
# =============================================================================

def _overlay_mean_of_means(
    fig, theta_bar, *, color=None, ls="-", lw=0.5, ms=3.5,
    label=None, add_legend=True, legend_bbox_to_anchor=(1.5, 0.8),
):
    """Mark θ̄: vertical line on 1D diagonal; circle on 2D panels (no Truth crosshairs)."""
    if fig is None:
        return
    if color is None:
        color = COLOR_RECENTERED
    if label is None:
        label = LABEL_MEAN_OF_MEANS_A
    theta_bar = np.asarray(theta_bar, dtype=float).reshape(-1)
    axes_grid = {}
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        if spec is None:
            continue
        r0, r1 = spec.rowspan.start, spec.rowspan.stop
        c0, c1 = spec.colspan.start, spec.colspan.stop
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        axes_grid[(r0, c0)] = ax
    if not axes_grid:
        return
    n_ax = max(r for r, _ in axes_grid) + 1
    for i in range(n_ax):
        for j in range(i + 1):
            ax = axes_grid.get((i, j))
            if ax is None or j >= len(theta_bar) or i >= len(theta_bar):
                continue
            if i == j:
                ax.axvline(theta_bar[j], color=color, ls=ls, lw=lw, zorder=100)
            else:
                ax.plot(
                    theta_bar[j], theta_bar[i],
                    linestyle="none", marker="o", markersize=ms,
                    color=color, markeredgecolor=color, zorder=100,
                )

    if not add_legend:
        return
    handle = Line2D(
        [0], [0],
        color=color, ls=ls, lw=lw,
        marker="o", markersize=ms,
        markerfacecolor=color, markeredgecolor=color,
    )
    ax_leg = next((ax for ax in fig.axes if ax.get_legend() is not None), None)
    if ax_leg is None:
        fig.legend(
            [handle], [label],
            loc="upper left", frameon=False, fontsize=10,
        )
        return
    old = ax_leg.get_legend()
    handles = list(old.legend_handles) + [handle]
    labels = [t.get_text() for t in old.get_texts()] + [label]
    keep = [(h, lab) for h, lab in zip(handles, labels) if lab and not lab.startswith("_")]
    if not keep:
        return
    handles, labels = zip(*keep)
    old.remove()
    leg_kwargs = dict(loc="upper left", fontsize=10, frameon=False)
    if legend_bbox_to_anchor is not None:
        leg_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
    ax_leg.legend(handles, labels, **leg_kwargs)


def plot_figA_mean_vs_indiv_means_contours(
    save_name="figA_mean_vs_indiv_means_contours",
):
    """A1 style with K=3 ensemble: CV-mean mix + grey rand + brown recentered + θ̄."""
    statistics_row = list(STATISTICS_FULL_ROW)
    tag_masks_row = TAG_MASK_FULL
    color_mean = COLORS_FID[3]
    loc_leg = (1.5, 0.8)

    # Ensemble CV-mean mixture (colored contour).
    mix_rp, missing_mean = load_ensemble_member_samples(
        statistics_row, tag_masks_row,
        test_mode="cvmean", k_members=N_ENSEMBLE_K, n_total=N_ENSEMBLE_DRAWS,
    )
    if mix_rp is None:
        raise FileNotFoundError(
            "ensemble CV-mean missing: " + "; ".join(missing_mean)
        )
    fn_pn = model_dir(statistics_row, tag_masks_row, 0) / "param_names.txt"
    names_rp = list(np.loadtxt(fn_pn, dtype=str))
    mix_u, names_u = utils_inference.unreparameterize_theta(mix_rp, names_rp, strict=False)
    names_u = list(names_u)
    i_zoom = [names_u.index(pn) for pn in PARAM_NAMES_ZOOM]
    mean_samples = mix_u[:, i_zoom] if mix_u.ndim == 2 else mix_u[:, 0, i_zoom]

    def _build_a1_ensemble_recentered():
        """Stack K cvindiv chains → recenter pooled posterior."""
        samples_3d, missing = load_ensemble_member_samples(
            statistics_row, tag_masks_row,
            test_mode="cvindiv", k_members=N_ENSEMBLE_K,
        )
        if samples_3d is None:
            raise FileNotFoundError(
                "ensemble cvindiv missing: " + "; ".join(missing)
            )
        samples_u, names_su = utils_inference.unreparameterize_theta(
            samples_3d, names_rp, strict=False,
        )
        names_su = list(names_su)
        i_z = [names_su.index(pn) for pn in PARAM_NAMES_ZOOM]
        samples_zoom = samples_u[:, :, i_z]
        theta_hat = np.nanmean(samples_zoom, axis=0)
        finite_obs = np.all(np.isfinite(theta_hat), axis=1)
        theta_hat = theta_hat[finite_obs]
        samples_zoom_ok = samples_zoom[:, finite_obs, :]
        n_obs_ok = theta_hat.shape[0]
        theta_bar = np.mean(theta_hat, axis=0)
        n_draw = samples_zoom_ok.shape[0]
        k = min(int(K_PER_OBS), n_draw)
        draw_idx = np.random.default_rng(1).choice(n_draw, size=k, replace=False)
        s_tilde = (
            samples_zoom_ok[draw_idx]
            - theta_hat[None, :, :]
            + theta_bar[None, None, :]
        )
        pooled_recentered = s_tilde.reshape(k * n_obs_ok, len(PARAM_NAMES_ZOOM))
        return dict(
            samples_zoom=samples_zoom_ok,
            theta_hat=theta_hat,
            theta_bar=theta_bar,
            pooled_recentered=pooled_recentered,
            k=k,
        )

    _a1 = load_or_build(
        f"figA1_ensK{N_ENSEMBLE_K}_recentered_pk_bispec_pgm_k{K_PER_OBS}",
        _build_a1_ensemble_recentered,
    )
    samples_zoom = _a1["samples_zoom"]
    theta_bar = _a1["theta_bar"]
    pooled_recentered = _a1["pooled_recentered"]
    print(
        f"ensemble A1 (K={N_ENSEMBLE_K}): "
        f"n_obs={samples_zoom.shape[1]}, "
        f"pooled recentered={pooled_recentered.shape[0]} "
        f"(× k={_a1['k']} draws/obs)"
    )

    # Thin grey random individual posteriors from stacked ensemble.
    n_obs_avail = samples_zoom.shape[1]
    obs_idx = np.random.default_rng(42).choice(
        n_obs_avail, size=min(N_RAND_A, n_obs_avail), replace=False,
    )
    rand_samples = [samples_zoom[:, j, :] for j in obs_idx]
    n_extra = len(rand_samples)
    print(f"overlaying {n_extra} random indiv. posteriors, obs_idx={obs_idx}")

    _, _, _, _, _, _, theta_show, _ = setup_fixed_mean_test()

    colors = [color_mean] + [COLOR_RAND] * n_extra + [COLOR_RECENTERED]
    labels = (
        [LABEL_MEAN_A, LABEL_RAND_A]
        + [f"_rand{i}" for i in range(1, n_extra)]
        + [LABEL_RECENTERED_A]
    )
    shades = [True] + [False] * n_extra + [False]
    linewidths = [LW_MEAN_A] + [LW_RAND_A] * n_extra + [LW_RECENTERED_A]
    smooths = [4] + [4] * n_extra + [4]
    bins_list = [8] + [8] * n_extra + [8]
    kdes = [False] + [False] * n_extra + [False]
    show_label_in_legend = [True, True] + [False] * (n_extra - 1) + [True]
    samples_list = (
        [(mean_samples, list(PARAM_NAMES_ZOOM))]
        + [(s, list(PARAM_NAMES_ZOOM)) for s in rand_samples]
        + [(pooled_recentered, list(PARAM_NAMES_ZOOM))]
    )
    n_contours = len(colors)
    tags_dummy = [f"_ensA{i}" for i in range(n_contours)]

    fig = plotter.plot_contours_inf(
        param_names=PARAM_NAMES_ZOOM, idx_obs=idx_obs, theta_obs_true=theta_show,
        inf_methods=["sbi"] * n_contours,
        tags_inf=tags_dummy,
        tags_test=[None] * n_contours,
        colors=colors,
        labels=labels,
        shades=shades,
        linewidths=linewidths,
        smooths=smooths,
        bins_list=bins_list,
        kdes=kdes,
        show_label_in_legend=show_label_in_legend,
        fontsize_legend=10,
        legend_location=None,
        legend_loc=None,
        loc_legend=loc_leg,
        title=None,
        extents=EXTENTS_A,
        figsize=FIGSIZE_CORNER,
        unreparameterize=False,  # already physical
        samples_list=samples_list,
        show=False,
    )
    _overlay_mean_of_means(
        fig, theta_bar, color=COLOR_RECENTERED, ls="-", lw=0.5,
        legend_bbox_to_anchor=loc_leg,
    )
    if fig is not None:
        for ax in fig.axes:
            ax.xaxis.label.set_fontsize(14)
            ax.yaxis.label.set_fontsize(14)
            ax.tick_params(labelsize=12)
            if ax.get_title():
                ax.title.set_fontsize(12)
        save_figure(fig, save_name)
    plt.show()


# =============================================================================
# Appendix B — full posteriors
# =============================================================================

def plot_figB1_cosmic_variance_full_posteriors(
    save_name="figB1_cosmic_variance_full_posteriors",
):
    tags_inf, labels, colors, tags_test, keep, theta_obs_full, _, param_vary = setup_fixed_mean_test()
    samples_list, labels_e, colors_e, tags_e, _ = load_fiducial_ensemble_samples_list(test_mode="cvmean")
    idxs = list(range(len(samples_list)))
    extents_all = {
        **genp.get_bounds("cosmo"), **genp.get_bounds("bias"),
        **genp.get_bounds("Anoise", anoise_option=ANOISE_OPTION),
    }
    extents_all["b1"] = extents_contours["b1"]
    # Legend placement matches 6-11 paper notebook: loc_legend=(-3, 0.9).
    fig = plotter.plot_contours_inf(
        param_names=list(param_vary), idx_obs=idx_obs,
        theta_obs_true=np.asarray(theta_obs_full, dtype=float).reshape(-1),
        inf_methods=["sbi"] * len(idxs),
        tags_inf=tags_e, tags_test=[""] * len(idxs),
        colors=colors_e, labels=labels_e,
        title=None, extents=extents_all, figsize=FIGSIZE_ALL_PARAMS,
        fontsize_legend=16,
        legend_location=None,
        legend_loc=None,
        loc_legend=(-3, 0.9),
        unreparameterize=True,
        samples_list=samples_list,
        show=False,
    )
    if fig is not None:
        save_figure(fig, save_name)
    plt.show()


def plot_figB2_shame_full_posteriors(save_name="figB2_shame_full_posteriors"):
    tags_inf, labels, colors, tags_test, keep, param_vary, _ = setup_shame_test()
    theta_dict = data_loader.load_params_ood(DATA_MODE_TEST_SHAME, TAG_MOCK_SHAME)
    theta_obs_full = np.array([theta_dict.get(pn, np.nan) for pn in param_vary])
    samples_list, labels_e, colors_e, tags_e, _ = load_fiducial_ensemble_samples_list(
        test_mode="shame", tag_mock=TAG_MOCK_SHAME,
    )
    extents_all = {
        **genp.get_bounds("cosmo"), **genp.get_bounds("bias"),
        **genp.get_bounds("Anoise", anoise_option=ANOISE_OPTION),
    }
    extents_all["b1"] = extents_contours["b1"]
    # Truth for all params except Laplacian bias (b_lap / bl); keep b1, b2, bs2.
    truth_exclude = {"bl"}
    truth_loc = {
        pn: float(theta_obs_full[param_vary.index(pn)])
        for pn in param_vary
        if pn not in truth_exclude and np.isfinite(theta_obs_full[param_vary.index(pn)])
    }
    # Legend placement matches 6-11 paper notebook: loc_legend=(-3, 0.9).
    fig = plotter.plot_contours_inf(
        param_names=list(param_vary), idx_obs=idx_obs,
        theta_obs_true=None,
        add_truth=True,
        truth_locations=[truth_loc],
        truth_colors=["k"],
        inf_methods=["sbi"] * len(samples_list),
        tags_inf=tags_e, tags_test=[""] * len(samples_list),
        colors=colors_e, labels=labels_e,
        title=None, extents=extents_all, figsize=FIGSIZE_ALL_PARAMS,
        fontsize_legend=16,
        legend_location=None,
        legend_loc=None,
        loc_legend=(-3, 0.9),
        unreparameterize=True,
        samples_list=samples_list,
        show=False,
    )
    if fig is not None:
        save_figure(fig, save_name)
    plt.show()
