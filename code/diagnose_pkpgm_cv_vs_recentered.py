#!/usr/bin/env python3
"""Diagnostics for CV-mean vs recentered-pooled disagreement (esp. pk+pgm).

Writes JSON under figures/2026-08-19_figure_variants/.
Run with benv (needs sbi for NPE log_prob / kmax sampling):
  conda run -n benv python code/diagnose_pkpgm_cv_vs_recentered.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats as spstats

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import data_loader  # noqa: E402
import paths  # noqa: E402
import utils_inference  # noqa: E402
import utils_plot  # noqa: E402
from ensemble_kmax_utils import parse_mask_to_tags_mask  # noqa: E402

OUT_DIR = REPO_ROOT / "figures" / "2026-08-19_figure_variants"
CACHE_DIR = REPO_ROOT / "figures" / "paper_figures_cache"
DIR_SBI = paths.DIR_RESULTS / "results_sbi"

BX = 32
N_TRAIN = 10000
TAG_INF_BEST_SUFFIX = "_best-rand30"
DATA_MODE = "muchisimocks"
TAG_PARAMS_TRAIN = "_p5_n10000"
TAG_BIASPARAMS_TRAIN = "_biasnoisenest_p9_n320000"
TAG_NOISE_TRAIN = "_noise_unit_p5_n10000"
TAG_REPARAM = "_rp"
TAG_PARAMS_TEST = "_shame_p0_n1000"
TAG_BIASPARAMS_TEST = "_biasshame_noisebest_p0_n1"
TAG_NOISE_TEST = "_noise_unit_shame_p0_n1000"
PARAM_NAMES = ["omega_cold", "sigma8_cold", "b1"]

FID_COMBOS = [
    (["pk"], "", "Pgg"),
    (["pk", "pgm"], "_kpgm0.25", "Pgg+Pgm"),
    (["pk", "bispec"], "_kb0.25", "Pgg+Bggg"),
    (["pk", "bispec", "pgm"], "_kb0.25_kpgm0.25", "Pgg+Bggg+Pgm"),
    (["pgm"], "_kpgm0.25", "Pgm"),
    (["bispec"], "_kb0.25", "Bggg"),
]

KPGM_TAGS = ["_kpgm0.1", "_kpgm0.15", "_kpgm0.2", "_kpgm0.25", "_kpgm0.3", "_kpgm0.35", ""]
# Only use saved individual samples for RC/CV ratios (no on-the-fly NPE sampling)
KPGM_SAMPLE_INDIV = set()  # empty: never NPE-sample; saved indiv only


def err68(x, axis=None):
    p16, p84 = np.nanpercentile(x, [16, 84], axis=axis)
    return 0.5 * (p84 - p16)


def cache_name(statistics_row):
    key = "_".join(statistics_row)
    prefix = "figA1b" if statistics_row == ["pk", "bispec", "pgm"] else "figA1"
    return f"{prefix}_recentered_pooled_{key}_k500"


def sample_path(tag_inf, tag_test):
    return DIR_SBI / f"sbi{tag_inf}" / f"samples_test{tag_test}_pred.npy"


def setup_tags(statistics_row, mask):
    tags_inf, _, _, _ = utils_plot.setup_inference_tags(
        data_mode=DATA_MODE,
        tag_params=TAG_PARAMS_TRAIN,
        tag_biasparams=TAG_BIASPARAMS_TRAIN,
        statistics_arr=[statistics_row],
        bx=BX,
        tag_noise=TAG_NOISE_TRAIN,
        tag_reparam=TAG_REPARAM,
        n_train=N_TRAIN,
        tags_mask=[mask],
    )
    tag_inf = tags_inf[0] + TAG_INF_BEST_SUFFIX
    tag_stats = f'_{"_".join(statistics_row)}'
    tag_mean = utils_plot.setup_test_tags(
        data_mode=DATA_MODE,
        tag_params_test=TAG_PARAMS_TEST,
        tags_biasparams_test=TAG_BIASPARAMS_TEST,
        tag_stats_arr=[tag_stats],
        tag_noise_test=TAG_NOISE_TEST,
        tag_datagen_test="_mean",
        tags_mask_test=[mask],
    )[0]
    tag_samp = utils_plot.setup_test_tags(
        data_mode=DATA_MODE,
        tag_params_test=TAG_PARAMS_TEST,
        tags_biasparams_test=TAG_BIASPARAMS_TEST,
        tag_stats_arr=[tag_stats],
        tag_noise_test=TAG_NOISE_TEST,
        tag_datagen_test="",
        tags_mask_test=[mask],
    )[0]
    return tag_inf, tag_mean, tag_samp


def unreparam_zoom(samples_2d, names):
    s_u, names_u = utils_inference.unreparameterize_theta(
        samples_2d, list(names), strict=False
    )
    names_u = list(names_u)
    i = [names_u.index(pn) for pn in PARAM_NAMES]
    return np.asarray(s_u[:, i], dtype=float)


def subsample_rows(X, n_max=80_000, seed=0):
    X = np.asarray(X, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] > n_max:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(X.shape[0], size=n_max, replace=False)]
    return X


def mahalanobis2(X, mu=None, cov=None):
    X = np.asarray(X, dtype=float)
    if mu is None:
        mu = np.mean(X, axis=0)
    if cov is None:
        cov = np.cov(X, rowvar=False)
    d = X.shape[1]
    cov = cov + 1e-12 * np.eye(d)
    prec = np.linalg.inv(cov)
    delta = X - mu
    return np.einsum("ni,ij,nj->n", delta, prec, delta), mu, cov


def joint_gaussianity(X):
    """Cheap multi-D normality: Mahalanobis^2 vs chi2_d (+ Mardia on subsample)."""
    X = subsample_rows(X, n_max=80_000, seed=0)
    n, d = X.shape
    m2, _, _ = mahalanobis2(X)
    q68 = float(spstats.chi2.ppf(0.682689492137, d))
    q95 = float(spstats.chi2.ppf(0.954499736104, d))
    frac68 = float(np.mean(m2 <= q68))
    frac95 = float(np.mean(m2 <= q95))
    ks_stat, ks_p = spstats.kstest(m2, "chi2", args=(d,))

    # Mardia on at most 5k points (avoid n×n)
    Xs = subsample_rows(X, n_max=5000, seed=1)
    ns, ds = Xs.shape
    mu = np.mean(Xs, axis=0)
    C = np.cov(Xs, rowvar=False) * (ns - 1) / ns + 1e-12 * np.eye(ds)
    prec = np.linalg.inv(C)
    D = Xs - mu
    # A_ii = mahalanobis^2 with 1/n cov
    a_diag = np.einsum("ni,ij,nj->n", D, prec, D)
    # skew: (1/n^2) sum_{i,j} A_ij^3 — compute via chunks
    # A = D @ prec @ D.T
    Z = D @ prec  # (n,d)
    # A_ij = Z_i · D_j
    b1d = 0.0
    chunk = 500
    for i0 in range(0, ns, chunk):
        A_block = Z[i0 : i0 + chunk] @ D.T  # (chunk, n)
        b1d += float(np.sum(A_block**3))
    b1d /= ns**2
    b2d = float(np.mean(a_diag**2))
    df_skew = ds * (ds + 1) * (ds + 2) / 6.0
    skew_stat = ns * b1d / 6.0
    skew_p = float(1.0 - spstats.chi2.cdf(skew_stat, df_skew))
    expected_kurt = ds * (ds + 2)
    kurt_z = (b2d - expected_kurt) / np.sqrt(8.0 * ds * (ds + 2) / ns)

    return {
        "n": int(n),
        "d": int(d),
        "frac_within_chi2_68": frac68,
        "frac_within_chi2_95": frac95,
        "delta_frac_68": frac68 - 0.682689492137,
        "delta_frac_95": frac95 - 0.954499736104,
        "ks_m2_chi2": float(ks_stat),
        "ks_p": float(ks_p),
        "mean_m2": float(np.mean(m2)),
        "expected_mean_m2": float(d),
        "mardia_skew": float(b1d),
        "mardia_skew_stat": float(skew_stat),
        "mardia_skew_p": skew_p,
        "mardia_kurt": float(b2d),
        "mardia_kurt_expected": float(expected_kurt),
        "mardia_kurt_z": float(kurt_z),
    }


def load_cv_rc(statistics_row, mask):
    tag_inf, tag_mean, tag_samp = setup_tags(statistics_row, mask)
    names = np.loadtxt(DIR_SBI / f"sbi{tag_inf}" / "param_names.txt", dtype=str)
    sm = np.load(sample_path(tag_inf, tag_mean))
    sm = sm[:, 0, :] if sm.ndim == 3 else sm
    cv = unreparam_zoom(sm, names)
    with open(CACHE_DIR / f"{cache_name(statistics_row)}.pkl", "rb") as f:
        bundle = pickle.load(f)
    return cv, bundle["pooled_recentered"], bundle["theta_hat"], tag_inf, tag_mean, tag_samp, names


def concat_scaled_y(y_list, scalers):
    """y_list: list of (n_obs, n_bins) unscaled; return (n_obs, n_feat) scaled."""
    parts = []
    for y, sc in zip(y_list, scalers):
        parts.append(sc.scale(np.asarray(y, dtype=float)))
    return np.concatenate(parts, axis=1)


def load_scalers(tag_inf, statistics):
    scalers = []
    for stat in statistics:
        with open(DIR_SBI / f"sbi{tag_inf}" / f"scaler_y_{stat}.p", "rb") as f:
            scalers.append(pickle.load(f))
    return scalers


def load_test_y(statistics, mask):
    tags = parse_mask_to_tags_mask(statistics, mask)
    k, y, y_err, *_rest = data_loader.load_data(
        DATA_MODE,
        statistics,
        TAG_PARAMS_TEST,
        TAG_BIASPARAMS_TEST,
        tag_noise=TAG_NOISE_TEST,
        tags_mask=tags,
    )
    return y


def feature_space_diagnostics(statistics, mask, tag_inf):
    """Data-space structure of the 1000 fixed-cosmo mocks vs their mean.

    The mean is the centroid by construction, so distance-to-mean-in-cloud is
    uninformative. Instead we measure:
      - mean vs median offset (skewed sampling → Jensen effects)
      - feature skewness/kurtosis of the noise
      - pk–pgm noise correlation (shared cosmic variance), when both present
    """
    scalers = load_scalers(tag_inf, statistics)
    y_test = load_test_y(statistics, mask)
    X_test = concat_scaled_y(y_test, scalers)  # (n_obs, n_feat)
    x_bar = np.mean(X_test, axis=0)
    x_med = np.median(X_test, axis=0)
    n_obs, n_feat = X_test.shape

    sig = X_test.std(axis=0) + 1e-12
    # mean–median offset in noise units
    z_mm = (x_bar - x_med) / sig
    euc_mean_vs_median = float(np.linalg.norm(z_mm))
    # typical individual distance to median
    z_indiv_med = (X_test - x_med) / sig
    euc_indiv_vs_median = np.linalg.norm(z_indiv_med, axis=1)
    # expected for symmetric noise: mean≈median so euc_mean_vs_median≈0

    # Feature skewness / kurtosis of sampling distribution
    rng = np.random.default_rng(0)
    feat_idx = rng.choice(n_feat, size=min(64, n_feat), replace=False)
    feat_skew = spstats.skew(X_test[:, feat_idx], axis=0, nan_policy="omit")
    feat_kurt = spstats.kurtosis(X_test[:, feat_idx], axis=0, nan_policy="omit")

    # Per-stat mean–median and skew
    dims = [sc.scale(np.asarray(y_test[i][:1], dtype=float)).shape[1] for i, sc in enumerate(scalers)]
    sl = 0
    stat_mean_med = {}
    stat_skew = {}
    for stat, d in zip(statistics, dims):
        zb = z_mm[sl : sl + d]
        stat_mean_med[stat] = float(np.linalg.norm(zb))
        sk = spstats.skew(X_test[:, sl : sl + d], axis=0, nan_policy="omit")
        stat_skew[stat] = float(np.mean(np.abs(sk)))
        sl += d

    # Cross-stat noise correlation: mean |corr| between pk and pgm features
    cross_corr_abs_mean = None
    if "pk" in statistics and "pgm" in statistics:
        i_pk = statistics.index("pk")
        i_pgm = statistics.index("pgm")
        # unscaled y for physical correlation of residuals
        ypk = np.asarray(y_test[i_pk], dtype=float)
        ypgm = np.asarray(y_test[i_pgm], dtype=float)
        rpk = ypk - ypk.mean(0)
        rpgm = ypgm - ypgm.mean(0)
        # correlation matrix between residual vectors (flatten pair via bin-wise corr)
        n_pair = min(rpk.shape[1], rpgm.shape[1], 28)
        corrs = []
        for a in range(n_pair):
            for b in range(n_pair):
                c = np.corrcoef(rpk[:, a], rpgm[:, b])[0, 1]
                if np.isfinite(c):
                    corrs.append(abs(c))
        cross_corr_abs_mean = float(np.mean(corrs)) if corrs else None
        # same-k approx: corr of bin i with bin i
        same = []
        for a in range(n_pair):
            c = np.corrcoef(rpk[:, a], rpgm[:, a])[0, 1]
            if np.isfinite(c):
                same.append(c)
        cross_corr_samek_mean = float(np.mean(same)) if same else None
    else:
        cross_corr_samek_mean = None

    return {
        "n_obs": int(n_obs),
        "n_feat": int(n_feat),
        "euc_mean_vs_median": euc_mean_vs_median,
        "euc_indiv_vs_median_median": float(np.median(euc_indiv_vs_median)),
        "mean_med_over_indiv_med": euc_mean_vs_median
        / float(np.median(euc_indiv_vs_median) + 1e-12),
        "stat_euc_mean_vs_median": stat_mean_med,
        "stat_abs_skew_mean": stat_skew,
        "feat_skew_abs_mean": float(np.mean(np.abs(feat_skew))),
        "feat_kurt_abs_mean": float(np.mean(np.abs(feat_kurt))),
        "feat_skew_abs_p90": float(np.percentile(np.abs(feat_skew), 90)),
        "feat_kurt_abs_p90": float(np.percentile(np.abs(feat_kurt), 90)),
        "pk_pgm_crosscorr_abs_mean": cross_corr_abs_mean,
        "pk_pgm_samek_corr_mean": cross_corr_samek_mean,
    }


def load_posterior(tag_inf):
    with open(DIR_SBI / f"sbi{tag_inf}" / "posterior.p", "rb") as f:
        return pickle.load(f)


def theta_true_train_coords(tag_inf):
    """Shame truth in NPE training coordinates (reparameterized), from fixed dicts."""
    names = list(np.loadtxt(DIR_SBI / f"sbi{tag_inf}" / "param_names.txt", dtype=str))
    _, cosmo_fixed, _, bias_fixed, *_ = data_loader.load_params(
        TAG_PARAMS_TEST, TAG_BIASPARAMS_TEST
    )
    s8 = float(cosmo_fixed["sigma8_cold"])
    phys = {
        "omega_cold": float(cosmo_fixed["omega_cold"]),
        "sigma8_cold": s8,
        "b1": float(bias_fixed["b1"]),
        "b2": float(bias_fixed["b2"]),
        "bs2": float(bias_fixed["bs2"]),
        "bl": float(bias_fixed["bl"]),
        "An_homog": float(bias_fixed["An_homog"]),
        "An_b1": float(bias_fixed["An_b1"]),
        "An_b2": float(bias_fixed["An_b2"]),
        "An_bs2": float(bias_fixed["An_bs2"]),
        "An_bl": float(bias_fixed["An_bl"]),
    }
    # Match utils_inference.reparameterize_theta product conventions
    rp = {
        "omega_cold": phys["omega_cold"],
        "sigma8_cold": phys["sigma8_cold"],
        "sigma8_cold_x_b1": s8 * phys["b1"],
        "sigma8_cold_sq_x_b2": (s8**2) * phys["b2"],
        "sigma8_cold_sq_x_bs2": (s8**2) * phys["bs2"],
        "sigma8_cold_x_bl": s8 * phys["bl"],
        "An_homog": phys["An_homog"],
        "sigma8_cold_x_An_b1": s8 * phys["An_b1"],
        "sigma8_cold_sq_x_An_b2": (s8**2) * phys["An_b2"],
        "sigma8_cold_sq_x_An_bs2": (s8**2) * phys["An_bs2"],
        "sigma8_cold_x_An_bl": s8 * phys["An_bl"],
    }
    cols = [rp[pn] if pn in rp else phys[pn] for pn in names]
    return np.asarray(cols, dtype=np.float32), names


def npe_score_diagnostics(statistics, mask, tag_inf, n_obs_sub=100):
    import torch

    posterior = load_posterior(tag_inf)
    scalers = load_scalers(tag_inf, statistics)
    y_test = load_test_y(statistics, mask)
    X = concat_scaled_y(y_test, scalers).astype(np.float32)
    x_bar = X.mean(0).astype(np.float32)

    theta_true, _ = theta_true_train_coords(tag_inf)
    theta_t = torch.as_tensor(theta_true[None, :])

    def logp(x_np):
        x_arr = np.asarray(x_np, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        n = x_arr.shape[0]
        # Evaluate one observation at a time (sbi batching APIs vary by version)
        out = np.empty(n, dtype=float)
        with torch.no_grad():
            for i in range(n):
                x_t = torch.as_tensor(x_arr[i : i + 1])
                lp = posterior.log_prob(theta_t, x=x_t)
                out[i] = float(lp.detach().cpu().numpy().reshape(-1)[0])
        return out

    lp_bar = float(logp(x_bar)[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], size=min(n_obs_sub, X.shape[0]), replace=False)
    lp_indiv = logp(X[idx])

    # Entropy proxy: -E[log q(theta|x)] for a few draws from saved samples if present
    return {
        "logp_true_given_mean": lp_bar,
        "logp_true_given_indiv_mean": float(np.mean(lp_indiv)),
        "logp_true_given_indiv_median": float(np.median(lp_indiv)),
        "logp_true_given_indiv_std": float(np.std(lp_indiv)),
        "logp_bar_minus_indiv_mean": lp_bar - float(np.mean(lp_indiv)),
        "logp_bar_percentile_among_indiv": float(
            spstats.percentileofscore(lp_indiv, lp_bar)
        ),
        "n_indiv": int(len(lp_indiv)),
    }


def widths_from_samples(samples_zoom):
    return {pn: float(err68(samples_zoom[:, i])) for i, pn in enumerate(PARAM_NAMES)}


def kpgm_rc_cv_sweep(n_indiv_sample=30, n_draws=1500):
    """RC/CV width ratio vs k_pgm for pk+pgm.

    For kpgm0.25 use saved individual samples; for others sample a few obs with NPE.
    """
    import torch

    results = []
    for mask in KPGM_TAGS:
        stats = ["pk", "pgm"]
        try:
            tag_inf, tag_mean, tag_samp = setup_tags(stats, mask)
        except Exception as e:
            results.append({"mask": mask, "error": f"setup: {e}"})
            continue
        fn_mean = sample_path(tag_inf, tag_mean)
        if not fn_mean.is_file():
            results.append({"mask": mask, "error": f"missing mean samples {fn_mean.name}"})
            continue
        names = np.loadtxt(DIR_SBI / f"sbi{tag_inf}" / "param_names.txt", dtype=str)
        sm = np.load(fn_mean)
        sm = sm[:, 0, :] if sm.ndim == 3 else sm
        cv = unreparam_zoom(sm, names)
        err_cv = np.array([err68(cv[:, i]) for i in range(3)])
        vol_cv = float(np.sqrt(max(np.linalg.det(np.cov(cv, rowvar=False)), 0.0)))

        fn_indiv = sample_path(tag_inf, tag_samp)
        rng = np.random.default_rng(0)
        if fn_indiv.is_file():
            arr = np.load(fn_indiv, mmap_mode="r")
            obs_idx = rng.choice(arr.shape[1], size=min(n_indiv_sample, arr.shape[1]), replace=False)
            errs = []
            vols = []
            for j in obs_idx:
                X = unreparam_zoom(np.asarray(arr[:, j, :], dtype=float), names)
                errs.append([err68(X[:, k]) for k in range(3)])
                vols.append(np.sqrt(max(np.linalg.det(np.cov(X, rowvar=False)), 0.0)))
            source = "saved_indiv"
        elif mask not in KPGM_SAMPLE_INDIV:
            results.append(
                {
                    "mask": mask,
                    "kpgm": None if mask == "" else float(mask.replace("_kpgm", "")),
                    "err_cv": {pn: float(err_cv[i]) for i, pn in enumerate(PARAM_NAMES)},
                    "vol_cv": vol_cv,
                    "source_indiv": "skipped",
                    "note": "no saved indiv; not in KPGM_SAMPLE_INDIV",
                }
            )
            print(f"  kpgm={mask or 'full'}: CV only (vol={vol_cv:.3g})", flush=True)
            continue
        else:
            try:
                posterior = load_posterior(tag_inf)
                scalers = load_scalers(tag_inf, stats)
                y_test = load_test_y(stats, mask)
                X = concat_scaled_y(y_test, scalers).astype(np.float32)
                obs_idx = rng.choice(X.shape[0], size=min(n_indiv_sample, X.shape[0]), replace=False)
                x_batch = torch.as_tensor(X[obs_idx])
                with torch.no_grad():
                    samples = posterior.sample_batched((n_draws,), x=x_batch)
                s = samples.detach().cpu().numpy()
                if s.ndim == 2:
                    s = s[:, None, :]
                if s.shape[0] == len(obs_idx) and s.shape[1] == n_draws:
                    s = np.transpose(s, (1, 0, 2))
                errs, vols = [], []
                for j in range(s.shape[1]):
                    Xz = unreparam_zoom(s[:, j, :], names)
                    errs.append([err68(Xz[:, k]) for k in range(3)])
                    vols.append(np.sqrt(max(np.linalg.det(np.cov(Xz, rowvar=False)), 0.0)))
                source = "npe_sampled"
            except Exception as e:
                results.append(
                    {
                        "mask": mask,
                        "kpgm": None if mask == "" else float(mask.replace("_kpgm", "")),
                        "err_cv": {pn: float(err_cv[i]) for i, pn in enumerate(PARAM_NAMES)},
                        "vol_cv": vol_cv,
                        "error_indiv": str(e),
                    }
                )
                print(f"  kpgm={mask or 'full'}: indiv FAIL {e}", flush=True)
                continue

        errs = np.asarray(errs, dtype=float)
        vols = np.asarray(vols, dtype=float)
        err_indiv = errs.mean(0)
        results.append(
            {
                "mask": mask,
                "kpgm": None if mask == "" else float(mask.replace("_kpgm", "")),
                "source_indiv": source,
                "err_cv": {pn: float(err_cv[i]) for i, pn in enumerate(PARAM_NAMES)},
                "err_indiv_mean": {
                    pn: float(err_indiv[i]) for i, pn in enumerate(PARAM_NAMES)
                },
                "ratio_indiv_over_cv": {
                    pn: float(err_indiv[i] / err_cv[i]) for i, pn in enumerate(PARAM_NAMES)
                },
                "vol_cv": vol_cv,
                "vol_indiv_mean": float(np.mean(vols)),
                "vol_ratio": float(np.mean(vols) / vol_cv),
                "n_indiv": int(len(vols)),
            }
        )
        print(
            f"  kpgm={mask or 'full'}: vol_indiv/vol_cv={np.mean(vols)/vol_cv:.3f} "
            f"width ratios={[float(err_indiv[i]/err_cv[i]) for i in range(3)]} ({source})",
            flush=True,
        )
    return results


def main():
    # Unbuffered progress for long runs
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}

    # --- 1) Multi-D gaussianity (reuse if present) ---
    joint_path = OUT_DIR / "posterior_joint_gaussianity.json"
    if joint_path.is_file():
        print("=== Multi-D gaussianity (cached) ===")
        with open(joint_path) as f:
            joint_rows = json.load(f)
    else:
        print("=== Multi-D gaussianity ===")
        joint_rows = []
        for stats, mask, label in FID_COMBOS:
            print(f"  {label}", flush=True)
            cv, rc, theta_hat, tag_inf, tag_mean, tag_samp, names = load_cv_rc(stats, mask)
            g_cv = joint_gaussianity(cv)
            g_rc = joint_gaussianity(rc)
            g_means = joint_gaussianity(theta_hat)
            joint_rows.append(
                {"combo": label, "cv": g_cv, "rc": g_rc, "means_dist": g_means}
            )
            print(
                f"    CV Δfrac68={g_cv['delta_frac_68']:+.3f} KS={g_cv['ks_m2_chi2']:.3f} "
                f"Mardia skew={g_cv['mardia_skew']:.3f} kurt_z={g_cv['mardia_kurt_z']:+.1f}",
                flush=True,
            )
            print(
                f"    RC Δfrac68={g_rc['delta_frac_68']:+.3f} KS={g_rc['ks_m2_chi2']:.3f} "
                f"Mardia skew={g_rc['mardia_skew']:.3f} kurt_z={g_rc['mardia_kurt_z']:+.1f}",
                flush=True,
            )
        with open(joint_path, "w") as f:
            json.dump(joint_rows, f, indent=2)
    out["joint_gaussianity"] = joint_rows

    # --- 2) Feature-space: is mean atypical among mocks? ---
    print("\n=== Feature-space diagnostics ===", flush=True)
    feat_rows = []
    for stats, mask, label in FID_COMBOS:
        print(f"  {label}", flush=True)
        tag_inf, _, _ = setup_tags(stats, mask)
        try:
            feat = feature_space_diagnostics(stats, mask, tag_inf)
            feat["combo"] = label
            feat_rows.append(feat)
            print(
                f"    mean–med euc={feat['euc_mean_vs_median']:.3f} "
                f"(vs indiv med {feat['euc_indiv_vs_median_median']:.2f}, "
                f"ratio={feat['mean_med_over_indiv_med']:.3f}) "
                f"|skew|={feat['feat_skew_abs_mean']:.2f} "
                f"stat_mm={feat['stat_euc_mean_vs_median']} "
                f"pk–pgm samek ρ={feat.get('pk_pgm_samek_corr_mean')}",
                flush=True,
            )
        except Exception as e:
            print(f"    FAIL: {e}", flush=True)
            feat_rows.append({"combo": label, "error": str(e)})
    out["feature_space"] = feat_rows
    with open(OUT_DIR / "posterior_feature_space.json", "w") as f:
        json.dump(feat_rows, f, indent=2)

    # --- 3) NPE score at truth ---
    print("\n=== NPE log q(θ_true | d) ===", flush=True)
    npe_rows = []
    for stats, mask, label in FID_COMBOS:
        print(f"  {label}", flush=True)
        tag_inf, _, _ = setup_tags(stats, mask)
        try:
            npe = npe_score_diagnostics(stats, mask, tag_inf, n_obs_sub=120)
            npe["combo"] = label
            npe_rows.append(npe)
            print(
                f"    logp_mean={npe['logp_true_given_mean']:.2f} "
                f"indiv_mean={npe['logp_true_given_indiv_mean']:.2f} "
                f"Δ={npe['logp_bar_minus_indiv_mean']:+.2f} "
                f"pctile={npe['logp_bar_percentile_among_indiv']:.1f}",
                flush=True,
            )
        except Exception as e:
            print(f"    FAIL: {e}", flush=True)
            npe_rows.append({"combo": label, "error": str(e)})
    out["npe_scores"] = npe_rows
    with open(OUT_DIR / "posterior_npe_scores.json", "w") as f:
        json.dump(npe_rows, f, indent=2)

    # --- 4) k_pgm sweep ---
    print("\n=== k_pgm RC/CV sweep (pk+pgm) ===", flush=True)
    kpgm_rows = kpgm_rc_cv_sweep(n_indiv_sample=20, n_draws=1000)
    out["kpgm_sweep"] = kpgm_rows
    with open(OUT_DIR / "posterior_kpgm_rc_cv_sweep.json", "w") as f:
        json.dump(kpgm_rows, f, indent=2)

    with open(OUT_DIR / "posterior_pkpgm_diagnostics_all.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote diagnostics under {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
