"""Functions moved out of code/plotter_figs.py in the v2 cleanup (2026-11-15).

No longer used by the pipeline; kept for reference / possible reuse, not maintained.
Imports are copied from code/plotter_figs.py; module-level names come from the star import.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from __future__ import annotations
import pickle
import sys
from pathlib import Path
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
import utils_model
import utils_inference
import utils_inference as ui
from ensemble_kmax_utils import (
    KP035_CONFIGS,
    N_ENSEMBLE_K,
    load_ensemble_member_samples,
    model_dir,
    member_samples_path,
    ensemble_coverage_members_exist,
    TAG_PARAMS_TRAIN,
    TAG_BIASPARAMS_TRAIN,
    TAG_NOISE_TRAIN,
    TAG_PARAMS_TEST_COV,
    TAG_BIASPARAMS_TEST_COV,
    TAG_NOISE_TEST_COV,
    TAG_PARAMS_TEST_FIXED,
    TAG_BIASPARAMS_TEST_FIXED,
    TAG_NOISE_TEST_FIXED,
    TAG_DATAGEN_TEST_MEAN,
)
from generate_config_inference import TAG_SWEEP
from plotter_figs import *  # noqa: F401,F403


def save_current_figure(name, ext=None):
    """Deprecated: prefer save_figure(fig, name) before plt.show()."""
    save_figure(plt.gcf(), name, ext=ext)


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


def compute_fom_full(cov):
    """FoM = 1/sqrt(det C) over all parameters in the chain covariance."""
    return float(ui.figure_of_merit(np.asarray(cov, dtype=float)))


def load_ensemble_fom3(statistics_row, tag_mask, *, test_mode="shame", tag_mock=TAG_MOCK_SHAME):
    samples, param_names, path_err = _ensemble_samples_for(
        statistics_row, tag_mask, test_mode=test_mode, tag_mock=tag_mock,
    )
    if samples is None:
        return None, path_err
    if samples.ndim == 2:
        samples = samples[:, np.newaxis, :]
    theta_pred, cov, param_names = ui.physical_moments_from_samples(samples[:, 0, :], param_names)
    _, fom3 = compute_fom_key_block(cov, param_names, theta_pred_row=theta_pred)
    if fom3 is None or not np.isfinite(fom3):
        return None, f"non-finite FoM3 (mask={tag_mask!r}, mode={test_mode})"
    return fom3, None


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


def build_tag_test_cv(statistics, tag_masks, *, mean=False):
    """Fixed-cosmo shame test tag (cosmic variance). mean=True → evaluate_mean (_mean)."""
    tag_stats = f"_{'_'.join(statistics)}"
    tag_mean = TAG_DATAGEN_TEST_MEAN if mean else ""
    return (
        f"_{data_mode}{tag_stats}{tag_masks}"
        f"{TAG_PARAMS_TEST_FIXED}{TAG_BIASPARAMS_TEST_FIXED}{TAG_NOISE_TEST_FIXED}"
        f"{tag_mean}"
    )


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

    pp_emp_arr = load_or_build("fig5_pp_emp_arr_kp035_ensK3_ip", _build_fig5_pp_emp)
    _plot_pp_overlay(
        pp_emp_arr, save_name,
        param_names_plot=param_names_plot,
        colors=colors_ok, labels=labels_abbrev, cred_levels=cred_levels,
    )


