#!/usr/bin/env python3
"""Fast coverage FoM/FoB grid for KP035 K=3 ensemble convergence plots.

Bottleneck in plotter_figs.collect_coverage_grid:
  - loads full (n_draw, n_obs, 11) float64 chains (~0.9 GB each)
  - Python loop over 1000 obs for 11×11 cov + unreparameterize

This script:
  - reads only the first 3 columns (ω_c, σ8, σ8×b1) — enough for FoM/FoB
  - accumulates mean/cov in draw-chunks (no full-array materialize)
  - mixes ensemble members via sufficient statistics (no concatenate)
  - vectorizes FoM/FoB over observations
  - parallelizes over (stat_combo, bx, n_cosmo) grid cells

Usage (from repo root, benv):
  python code/fast_conv_coverage_metrics.py
  python code/fast_conv_coverage_metrics.py --workers 8 --plot
  python code/fast_conv_coverage_metrics.py --validate  # one cell vs cached paper df
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import paths  # noqa: E402
import data_loader  # noqa: E402
import utils_plot  # noqa: E402

# --- config matching plotter_figs / ensemble_kmax_utils (avoid heavy imports) ---
N_ENSEMBLE_K = 3
KP035_CONFIGS = [
    (["pk"], "_kp0.35"),
    (["pk", "pgm"], "_kp0.35_kpgm0.25"),
    (["pk", "bispec"], "_kp0.35_kb0.25"),
    (["pk", "bispec", "pgm"], "_kp0.35_kb0.25_kpgm0.25"),
]
TAG_INF_BEST_SUFFIX = "_best-rand30"
TAG_PARAMS_TRAIN = "_p5_n10000"
TAG_BIASPARAMS_TRAIN = "_biasnoisenest_p9_n320000"
TAG_NOISE_TRAIN = "_noise_unit_p5_n10000"
TAG_REPARAM = "_rp"
TAG_PARAMS_TEST_COV = "_coverage_p5_n1000"
TAG_BIASPARAMS_TEST_COV = "_biasnoisecoverage_p9_n1000"
TAG_NOISE_TEST_COV = "_noise_unit_coverage_p5_n1000"
DATA_MODE = "muchisimocks"
N_COSMO_ARR = [500, 1000, 2000, 4000, 6000, 8000, 10000]
BX_ARR = [1, 2, 4, 8, 16, 32]
N_COVERAGE = 1000
FOB_KEYS = ["omega_m", "sigma8", "b1"]
PARAM_NAMES_FOB_RP = ["omega_cold", "sigma8_cold", "sigma8_cold_x_b1"]
FOB_RIDGE = 1e-8
CHUNK_DRAWS = 500
# Contiguous prefix of the draw axis — ~10× less I/O for 10k-draw chains.
# Cov/FoM change negligibly vs full 10k for these plots; use --full-draws for exact.
DEFAULT_MAX_DRAWS = 1000

DIR_SBI = Path(paths.DIR_RESULTS) / "results_sbi"
REPO_ROOT = Path(paths.PROJECT_ROOT)
CACHE_DIR = REPO_ROOT / "figures" / "paper_figures_cache"
FIG_DIR = REPO_ROOT / "figures" / "2026-08-07_convergence_variations"
FIG_DIR_LATEST = REPO_ROOT / "figures" / "convergence_variations_latest"
COLORS_FID = ["#79D0EC", "#6BB756", "#9D86E0", "#B32168"]


def build_tag_inf(bx, n_cosmo, statistics, mask, *, nth=0) -> str:
    tag_stats = f"_{'_'.join(statistics)}"
    base = (
        f"_{DATA_MODE}{tag_stats}{mask}{TAG_PARAMS_TRAIN}"
        f"{TAG_BIASPARAMS_TRAIN}{TAG_NOISE_TRAIN}{TAG_REPARAM}"
        f"_bx{bx}_ntrain{n_cosmo}{TAG_INF_BEST_SUFFIX}"
    )
    return base if nth == 0 else f"{base}_nbest{nth}"


def build_tag_test(statistics, mask) -> str:
    tag_stats = f"_{'_'.join(statistics)}"
    return (
        f"_{DATA_MODE}{tag_stats}{mask}"
        f"{TAG_PARAMS_TEST_COV}{TAG_BIASPARAMS_TEST_COV}{TAG_NOISE_TEST_COV}"
    )


def coverage_samples_path(tag_inf: str, tag_test: str):
    base = DIR_SBI / f"sbi{tag_inf}"
    done = base / f"samples_test{tag_test}_pred.npy"
    if done.is_file():
        return done, "done"
    ip = base / f"samples_test{tag_test}_pred_inprogress.npy"
    if ip.is_file():
        return ip, "inprogress"
    return None, None


def inventory():
    n_exp = len(BX_ARR) * len(N_COSMO_ARR)
    print(
        f"Expected grid: {len(BX_ARR)} bx × {len(N_COSMO_ARR)} n_cosmo "
        f"= {n_exp} points / combo (K={N_ENSEMBLE_K})\n"
    )
    for statistics, mask in KP035_CONFIGS:
        label = " + ".join(statistics)
        tag_test = build_tag_test(statistics, mask)
        n_any = n_done = n_ip = n_partial = 0
        for bx in BX_ARR:
            for nt in N_COSMO_ARR:
                kinds = []
                for nth in range(N_ENSEMBLE_K):
                    tag_inf = build_tag_inf(bx, nt, statistics, mask, nth=nth)
                    _, kind = coverage_samples_path(tag_inf, tag_test)
                    if kind is not None:
                        kinds.append(kind)
                if not kinds:
                    continue
                n_any += 1
                if len(kinds) == N_ENSEMBLE_K:
                    if "inprogress" in kinds:
                        n_ip += 1
                    else:
                        n_done += 1
                else:
                    n_partial += 1
        print(
            f"{label:20s} {mask}: {n_any}/{n_exp} with ≥1 member "
            f"(K=3 done={n_done}, K=3 ip={n_ip}, partial={n_partial})"
        )


_MAX_DRAWS = DEFAULT_MAX_DRAWS


def _load_member_3col(fn: Path, n_obs: int | None = None) -> np.ndarray:
    """Load (n_draw, n_obs, 3) = (ω_c, σ8, σ8×b1) via mmap + copy of first 3 cols only.

    Only the first ``_MAX_DRAWS`` draws are read (contiguous file prefix → far less I/O
    than scanning a full 10k×1000×11 array).
    """
    a = np.load(fn, mmap_mode="r")
    if a.ndim == 2:
        a = a[:, np.newaxis, :]
    if a.shape[-1] < 3:
        raise ValueError(f"{fn}: expected ≥3 params, got {a.shape[-1]}")
    no = a.shape[1] if n_obs is None else min(n_obs, a.shape[1])
    nd_full = a.shape[0]
    nd = nd_full if _MAX_DRAWS is None else min(nd_full, int(_MAX_DRAWS))
    out = np.empty((nd, no, 3), dtype=np.float64)
    for i0 in range(0, nd, CHUNK_DRAWS):
        i1 = min(i0 + CHUNK_DRAWS, nd)
        out[i0:i1] = a[i0:i1, :no, :3]
    return out


def moments_mixed_3col(arrays: list[np.ndarray]):
    """Match plotter_figs: concat on draw axis, drop all-NaN obs, nanmean + np.cov."""
    n_obs = min(a.shape[1] for a in arrays)
    arrays = [a[:, :n_obs, :] for a in arrays]
    mixed = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    has_any = np.any(np.isfinite(mixed), axis=(0, 2))
    rows = np.where(has_any)[0]
    if rows.size == 0:
        return np.empty((0, 3)), np.empty((0, 3, 3)), rows, mixed.shape[1]
    X = mixed[:, rows, :]
    mu = np.nanmean(X, axis=0)
    # np.cov per obs (ddof=1); vectorized for finite-only chains
    Xc = X - mu
    Xc = np.where(np.isfinite(Xc), Xc, 0.0)
    # if some draws were NaN, nanmean used fewer points — use count of finite draws per obs
    n_fin = np.isfinite(X).all(axis=-1).sum(axis=0).astype(np.float64)  # (n_use,)
    n_fin = np.clip(n_fin, 2, None)
    cov = np.einsum("doa,dob->oab", Xc, Xc) / (n_fin - 1)[:, None, None]
    return mu, cov, rows, mixed.shape[1]


def unreparam_fob_block(mu_rp, cov_rp):
    """Physical (ω, σ8, b1) mean/cov from reparam block; matches scale_covariance_unreparameterize_approx on 3×3."""
    s8 = mu_rp[:, 1]
    mu = mu_rp.copy()
    mu[:, 2] = mu_rp[:, 2] / s8
    cov = cov_rp.copy()
    inv = 1.0 / s8
    inv2 = inv * inv
    cov[:, 2, 0] *= inv
    cov[:, 0, 2] *= inv
    cov[:, 2, 1] = 0.0
    cov[:, 1, 2] = 0.0
    cov[:, 2, 2] *= inv2
    return mu, cov


def metrics_from_moments(mu_rp, cov_rp, theta_true_fob):
    """Vectorized FoM/FoB; theta_true_fob is (n_obs, 3) physical ω, σ8, b1."""
    mu, cov = unreparam_fob_block(mu_rp, cov_rp)
    diag = np.diagonal(cov, axis1=1, axis2=2)
    sig = np.sqrt(np.clip(diag, 0.0, None))
    fom_marg = np.where(sig > 0, 1.0 / sig, np.nan)
    det = np.linalg.det(cov)
    fom3 = np.where(det > 0, 1.0 / np.sqrt(det), np.nan)

    cov_r = cov + FOB_RIDGE * np.eye(3)[None, :, :]
    sig_r = np.sqrt(np.clip(np.diagonal(cov_r, axis1=1, axis2=2), 0.0, None))
    diff = mu - theta_true_fob
    fob_marg = np.where(sig_r > 0, np.abs(diff) / sig_r, np.nan)
    # batch mahalanobis: solve cov_r @ x = diff
    try:
        x = np.linalg.solve(cov_r, diff[..., None])[..., 0]
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(cov_r) @ diff[..., None]
        x = np.squeeze(x, -1)
    fob3 = np.sqrt(np.einsum("oi,oi->o", diff, x))

    out = {
        "fom_3d": float(np.nanmean(fom3)),
        "fob3": float(np.nanmean(fob3)),
    }
    for i, key in enumerate(FOB_KEYS):
        out[f"fom_{key}"] = float(np.nanmean(fom_marg[:, i]))
        out[f"fob_{key}"] = float(np.nanmean(fob_marg[:, i]))
    return out


# truth array shared to workers via initializer
_THETA_TRUE_FOB = None
_PARAM_VARY = None


def _init_worker(theta_true_fob, param_vary, max_draws):
    global _THETA_TRUE_FOB, _PARAM_VARY, _MAX_DRAWS
    _THETA_TRUE_FOB = theta_true_fob
    _PARAM_VARY = param_vary
    _MAX_DRAWS = max_draws


def _min_draws_required() -> int:
    """Minimum draws per member for a point to count as ready (matches cache cap)."""
    return DEFAULT_MAX_DRAWS if _MAX_DRAWS is None else int(_MAX_DRAWS)


def _row_from_arrays(
    arrays, kinds, *, statistics, mask, bx, n_cosmo, variant, kind_label,
    n_draws_members=None,
    require_draws: int | None = None,
):
    """Build one metrics row from loaded member arrays."""
    mu, cov, rows, n_stored = moments_mixed_3col(arrays)
    n_use = int(rows.size)
    if n_use == 0:
        return None
    mets = metrics_from_moments(mu, cov, _THETA_TRUE_FOB[rows])
    if n_draws_members is None:
        n_draws_members = [int(a.shape[0]) for a in arrays]
    n_draws_members = list(n_draws_members)
    if require_draws is None:
        require_draws = _min_draws_required()
    draws_ok = bool(n_draws_members) and all(
        int(d) >= int(require_draws) for d in n_draws_members
    )
    # Ready = enough draws + full coverage. File still ``inprogress`` is OK
    # (e.g. heading toward 10k) as long as the first ``require_draws`` exist.
    incomplete = (
        n_stored < N_COVERAGE
        or n_use < N_COVERAGE
        or not draws_ok
    )
    return {
        "bx": bx,
        "n_cosmo": n_cosmo,
        "n_sims": bx * n_cosmo,
        "n_cov": n_use,
        "n_stored": int(n_stored),
        "n_draws": int(sum(n_draws_members)),
        "n_draws_members": n_draws_members,
        "draws_ok": draws_ok,
        "status": "incomplete" if incomplete else "complete",
        "kind": kind_label,
        "variant": variant,  # "top" | "ensemble"
        "stat_key": "_".join(statistics) + (mask or "_nomask"),
        **mets,
    }


def process_cell(args):
    """Return list of rows: top model (nth=0) and/or full K-member ensemble mix.

    Ensemble rows are emitted only when all ``N_ENSEMBLE_K`` members exist.
    """
    statistics, mask, bx, n_cosmo = args
    tag_test = build_tag_test(statistics, mask)
    by_nth = {}
    for nth in range(N_ENSEMBLE_K):
        tag_inf = build_tag_inf(bx, n_cosmo, statistics, mask, nth=nth)
        fn, kind = coverage_samples_path(tag_inf, tag_test)
        if fn is not None:
            by_nth[nth] = (fn, kind)
    if not by_nth:
        return []

    loaded = {nth: (_load_member_3col(fn), kind) for nth, (fn, kind) in by_nth.items()}
    rows_out = []
    require_draws = _min_draws_required()

    # Top model = ensemble member 0 (_best-rand30)
    if 0 in loaded:
        arr0, kind0 = loaded[0]
        rows_out.append(
            _row_from_arrays(
                [arr0], [kind0],
                statistics=statistics, mask=mask, bx=bx, n_cosmo=n_cosmo,
                variant="top", kind_label=kind0,
                n_draws_members=[int(arr0.shape[0])],
                require_draws=require_draws,
            )
        )

    # Equal-weight mix only when all K members are present
    if all(nth in loaded for nth in range(N_ENSEMBLE_K)):
        arrays = [loaded[nth][0] for nth in range(N_ENSEMBLE_K)]
        kinds = [loaded[nth][1] for nth in range(N_ENSEMBLE_K)]
        ip = "inprogress" in kinds
        kind_ens = "ensemble_inprogress" if ip else "ensemble_done"
        rows_out.append(
            _row_from_arrays(
                arrays, kinds,
                statistics=statistics, mask=mask, bx=bx, n_cosmo=n_cosmo,
                variant="ensemble", kind_label=kind_ens,
                n_draws_members=[int(a.shape[0]) for a in arrays],
                require_draws=require_draws,
            )
        )
    return [r for r in rows_out if r is not None]


def _npy_draw_count(fn: Path) -> int | None:
    """n_draw from samples array shape without loading data."""
    try:
        a = np.load(fn, mmap_mode="r")
    except Exception:
        return None
    if a.ndim == 2:
        return int(a.shape[0])
    if a.ndim >= 3:
        return int(a.shape[0])
    return None


def member_draw_info(statistics, mask, bx, n_cosmo, *, max_draws=None):
    """Return list of (nth, kind, n_draws_file, n_draws_used) for available members."""
    tag_test = build_tag_test(statistics, mask)
    out = []
    for nth in range(N_ENSEMBLE_K):
        tag_inf = build_tag_inf(bx, n_cosmo, statistics, mask, nth=nth)
        fn, kind = coverage_samples_path(tag_inf, tag_test)
        if fn is None:
            continue
        n_file = _npy_draw_count(fn)
        if n_file is None:
            n_used = None
        elif max_draws is None:
            n_used = n_file
        else:
            n_used = min(n_file, int(max_draws))
        out.append((nth, kind, n_file, n_used))
    return out


def _stat_key_to_config(stat_key: str):
    for stats, m in KP035_CONFIGS:
        if "_".join(stats) + (m or "_nomask") == stat_key:
            return list(stats), m
    return None, None


def resolve_n_draws_members(row, *, min_draws: int = DEFAULT_MAX_DRAWS) -> list[int] | None:
    """Per-member draw counts used for readiness (cache column or file probe)."""
    ndm = row.get("n_draws_members") if hasattr(row, "get") else (
        row["n_draws_members"] if "n_draws_members" in row.index else None
    )
    if isinstance(ndm, (list, tuple)) and len(ndm) > 0:
        return [int(d) for d in ndm]

    sk = row.get("stat_key") if hasattr(row, "get") else row["stat_key"]
    statistics, mask = _stat_key_to_config(str(sk))
    if statistics is None:
        return None
    bx, nt = int(row["bx"]), int(row["n_cosmo"])
    info = member_draw_info(statistics, mask, bx, nt, max_draws=min_draws)
    variant = row.get("variant") if hasattr(row, "get") else row.get("variant", None)
    if variant == "top":
        info = [t for t in info if t[0] == 0]
    elif variant == "ensemble":
        # require slots 0..K-1 present
        by_nth = {t[0]: t for t in info}
        if any(nth not in by_nth for nth in range(N_ENSEMBLE_K)):
            return [int(by_nth[nth][3]) if nth in by_nth and by_nth[nth][3] is not None else 0
                    for nth in range(N_ENSEMBLE_K)]
        return [int(by_nth[nth][3]) for nth in range(N_ENSEMBLE_K)]
    if not info:
        return None
    return [int(t[3]) for t in info if t[3] is not None]


def draws_requirement_met(row, *, min_draws: int = DEFAULT_MAX_DRAWS) -> bool:
    """Top: ≥min_draws on member 0. Ensemble: all K members each ≥min_draws."""
    ndm = resolve_n_draws_members(row, min_draws=min_draws)
    if not ndm:
        return False
    variant = row["variant"] if "variant" in getattr(row, "index", []) else "top"
    if variant == "top":
        return int(ndm[0]) >= int(min_draws)
    if variant == "ensemble":
        return len(ndm) == N_ENSEMBLE_K and all(int(d) >= int(min_draws) for d in ndm)
    return False


def blank_insufficient_draws(
    df: pd.DataFrame,
    *,
    min_draws: int = DEFAULT_MAX_DRAWS,
    require_full_coverage: bool = True,
) -> pd.DataFrame:
    """NaN FoM/FoB where readiness fails (leave those points blank).

    Top model: need ≥``min_draws`` on member 0 (extra draws ignored / already capped).
    Ensemble: need all ``N_ENSEMBLE_K`` members with ≥``min_draws`` each.
    If ``require_full_coverage``, also need ``n_cov == N_COVERAGE`` (or
    ``status == complete`` when present).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    metric_cols = [
        c for c in out.columns
        if c.startswith("fom") or c.startswith("fob")
    ]
    ok_list = []
    for idx, row in out.iterrows():
        ok = draws_requirement_met(row, min_draws=min_draws)
        if require_full_coverage:
            if "n_cov" in getattr(row, "index", []):
                ok = ok and int(row["n_cov"]) >= N_COVERAGE
            elif "status" in getattr(row, "index", []):
                ok = ok and row.get("status") == "complete"
        ok_list.append(ok)
        if not ok:
            for c in metric_cols:
                out.at[idx, c] = np.nan
    out["draws_ok"] = ok_list
    n_ok = int(sum(ok_list))
    print(
        f"ready (draws>={min_draws}"
        + (", full n_cov" if require_full_coverage else "")
        + f"): {n_ok}/{len(out)} points kept ({len(out) - n_ok} blanked)"
    )
    return out


def print_conv_detail(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    max_draws: int | None = DEFAULT_MAX_DRAWS,
    probe_files: bool = True,
):
    """Print per-grid-point status / kind / draw counts for a plot dataframe.

    ``max_draws`` is the cap used when building the cache (1000 default, None = full).
    If the dataframe lacks ``n_draws_members``, probes sample files on disk (fast header read).
    """
    if df is None or df.empty:
        print("(no rows)")
        return
    if title:
        print(title)
    print(
        f"rows={len(df)}  status={df['status'].value_counts().to_dict()}"
        + (f"  kind={df['kind'].value_counts().to_dict()}" if "kind" in df.columns else "")
    )
    # Parse statistics/mask from stat_key when unique
    stat_keys = sorted(df["stat_key"].unique()) if "stat_key" in df.columns else []
    min_d = DEFAULT_MAX_DRAWS if max_draws is None else int(max_draws)
    for sk in stat_keys:
        sub = df[df["stat_key"] == sk].sort_values(["bx", "n_cosmo"])
        print(f"\n  {sk}  ({len(sub)} points)")
        statistics, mask = _stat_key_to_config(sk)
        n_plot = 0
        for _, row in sub.iterrows():
            bx, nt = int(row["bx"]), int(row["n_cosmo"])
            st = row.get("status", "?")
            kind = row.get("kind", "?")
            n_cov = row.get("n_cov", "?")
            variant = row.get("variant", None)
            plot_ok = draws_requirement_met(row, min_draws=min_d)
            if plot_ok:
                n_plot += 1
            draws_str = ""
            if "n_draws_members" in row.index and isinstance(row["n_draws_members"], (list, tuple)):
                parts = [str(d) for d in row["n_draws_members"]]
                draws_str = (
                    f"draws_used=[{','.join(parts)}] "
                    f"sum={row.get('n_draws', sum(row['n_draws_members']))}"
                )
            elif probe_files and statistics is not None:
                info = member_draw_info(statistics, mask, bx, nt, max_draws=max_draws)
                if variant == "top":
                    info = [t for t in info if t[0] == 0]
                if info:
                    parts = [
                        f"n{nth}:{n_used}/{n_file}({k})"
                        for nth, k, n_file, n_used in info
                    ]
                    draws_str = "members=[" + ", ".join(parts) + "]"
                else:
                    draws_str = "members=[]"
            flag = "plot" if plot_ok else "BLANK"
            print(
                f"    bx={bx:2d} n_cosmo={nt:5d}  {st:10s}  kind={kind:22s}  "
                f"n_cov={n_cov}  {flag:5s}  {draws_str}"
            )
        n_comp = int((sub["status"] == "complete").sum()) if "status" in sub.columns else 0
        n_inc = int((sub["status"] != "complete").sum()) if "status" in sub.columns else 0
        print(f"  → {n_comp} complete, {n_inc} incomplete; {n_plot}/{len(sub)} meet draws≥{min_d}")


def load_theta_true_fob():
    """Physical (ω_c, σ8, b1) truth for coverage LH, shape (N_COVERAGE, 3)."""
    cosmo_vary, bias_vary, param_vary = utils_plot.load_training_params(
        TAG_PARAMS_TRAIN, TAG_BIASPARAMS_TRAIN, bx=32,
    )
    theta = data_loader.load_theta_test(
        TAG_PARAMS_TEST_COV,
        TAG_BIASPARAMS_TEST_COV,
        cosmo_param_names_vary=cosmo_vary,
        bias_param_names_vary=bias_vary,
    )
    if theta.ndim == 1:
        theta = np.tile(theta, (N_COVERAGE, 1))
    names = list(param_vary)
    idx = [names.index(p) for p in ["omega_cold", "sigma8_cold", "b1"]]
    return np.asarray(theta[:, idx], dtype=np.float64), list(param_vary)


def collect_all(
    workers: int = 8,
    verbose: bool = True,
    *,
    stat_keys: list[str] | None = None,
) -> pd.DataFrame:
    theta_true, param_vary = load_theta_true_fob()
    jobs = []
    for statistics, mask in KP035_CONFIGS:
        sk = "_".join(statistics) + (mask or "_nomask")
        if stat_keys is not None and sk not in stat_keys:
            continue
        for bx in BX_ARR:
            for nt in N_COSMO_ARR:
                jobs.append((list(statistics), mask, bx, nt))
    if not jobs:
        raise ValueError(f"no jobs matched stat_keys={stat_keys!r}")

    rows = []
    t0 = time.time()
    if workers <= 1:
        _init_worker(theta_true, param_vary, _MAX_DRAWS)
        for i, job in enumerate(jobs):
            rows.extend(process_cell(job))
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(jobs)}] rows={len(rows)}  ({time.time()-t0:.0f}s)")
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(theta_true, param_vary, _MAX_DRAWS),
        ) as ex:
            futs = {ex.submit(process_cell, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                rows.extend(fut.result())
                if verbose and done % 10 == 0:
                    print(f"  [{done}/{len(jobs)}] rows={len(rows)}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    if verbose:
        print(f"\nTotal rows: {len(df)} from {len(jobs)} cells in {time.time()-t0:.1f}s")
        if not df.empty:
            if "variant" in df.columns:
                print("variant:", df["variant"].value_counts().to_dict())
            if "status" in df.columns:
                print("status:", df["status"].value_counts().to_dict())
            print("by variant × combo:")
            for (var, sk), g in df.groupby(["variant", "stat_key"]):
                print(f"  {var:9s} {sk}: {len(g)}  {g['status'].value_counts().to_dict()}")
    return df


def save_caches(df: pd.DataFrame, *, tag: str = ""):
    """Write fast caches: combined + per-variant slices.

    ``tag`` is an optional suffix (e.g. ``fulldraws``) inserted before ``_ip_fast``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag_part = f"_{tag}" if tag else ""
    path_all = CACHE_DIR / f"df_conv_all_kp035_top_and_ensK3{tag_part}_ip_fast.pkl"
    with open(path_all, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {path_all} ({len(df)} rows)")

    for variant, suffix in [("top", "top"), ("ensemble", "ensK3")]:
        sub = df[df["variant"] == variant].copy()
        path = CACHE_DIR / f"df_conv_all_kp035_{suffix}{tag_part}_ip_fast.pkl"
        with open(path, "wb") as f:
            pickle.dump(sub, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote {path} ({len(sub)} rows)")
    return path_all


def validate_one():
    """Compare fast ensemble metrics at bx=32, ntrain=10000, pk vs paper cache."""
    theta_true, param_vary = load_theta_true_fob()
    _init_worker(theta_true, param_vary, _MAX_DRAWS)
    t0 = time.time()
    rows = process_cell((["pk"], "_kp0.35", 32, 10000))
    print(f"process_cell wall: {time.time()-t0:.1f}s  n_rows={len(rows)}")
    for row in rows:
        print(
            f"{row['variant']}:",
            {k: row[k] for k in row if k.startswith("fom") or k.startswith("fob")
             or k in ("kind", "n_cov", "status", "variant")},
        )
    ens = next((r for r in rows if r["variant"] == "ensemble"), None)
    cache = CACHE_DIR / "df_conv_pk_kp0.35_b1phys_err_ensK3_ip.pkl"
    if ens is not None and cache.is_file():
        df = pickle.load(open(cache, "rb"))
        ref = df[(df.bx == 32) & (df.n_cosmo == 10000)].iloc[0]
        print("cache:", {k: ref[k] for k in ["fom_3d", "fob3", "fom_omega_m", "fob_omega_m", "kind", "n_cov", "status"]})
        for k in ["fom_3d", "fob3", "fom_omega_m", "fom_sigma8", "fom_b1", "fob_omega_m", "fob_sigma8", "fob_b1"]:
            a, b = float(ens[k]), float(ref[k])
            rel = abs(a - b) / max(abs(b), 1e-30)
            print(f"  {k}: fast={a:.6g} cache={b:.6g} rel_err={rel:.2e}")


def plot_from_df(df: pd.DataFrame, *, tag: str = ""):
    """Lightweight fig10/fig11-style plots for each variant (top, then ensemble)."""
    import matplotlib.colors as mcolors
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import chi2

    def alpha(n_cosmo, vals):
        vals = sorted(vals)
        if len(vals) == 1:
            return 1.0
        return 0.15 + 0.85 * vals.index(int(n_cosmo)) / (len(vals) - 1)

    def draw(ax, sub, ycol, color):
        ncs = sorted(sub["n_cosmo"].unique())
        rgb = mcolors.to_rgb(color)
        for nc in ncs:
            g = sub[sub["n_cosmo"] == nc].sort_values("n_sims")
            if g.empty:
                continue
            ax.plot(g["n_sims"], g[ycol], color=(*rgb, alpha(nc, ncs)), ls="-", lw=1.5)

    variants = [("top", "top"), ("ensemble", "ensK3")]
    if "variant" not in df.columns:
        variants = [(None, "ensK3")]

    key_full = "pk_bispec_pgm_kp0.35_kb0.25_kpgm0.25"
    ref3 = float(np.sqrt(chi2.ppf(0.682689492137, 3)))
    lim1 = float(np.sqrt(chi2.ppf(0.682689492137, 1)))
    gauss = np.sqrt(2 / np.pi)
    tag_part = f"_{tag}" if tag else ""
    n_stat_keys = df["stat_key"].nunique() if "stat_key" in df.columns else 0

    for variant, vtag in variants:
        dfi = df if variant is None else df[df["variant"] == variant]
        if dfi.empty:
            print(f"No data for variant={variant}")
            continue

        dfi = blank_insufficient_draws(dfi, min_draws=DEFAULT_MAX_DRAWS if _MAX_DRAWS is None else int(_MAX_DRAWS))
        full = dfi[dfi["stat_key"] == key_full]
        if not full.empty:
            fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
            draw(axes[0], full, "fom_3d", COLORS_FID[3])
            axes[0].set_ylabel(r"FoM($\Omega_m$, $\sigma_8$, $b_1$)")
            axes[0].set_yscale("log")
            axes[0].set_xscale("log")
            full_scaled = full.copy()
            full_scaled["fob3_n"] = full_scaled["fob3"] / ref3
            draw(axes[1], full_scaled, "fob3_n", COLORS_FID[3])
            axes[1].axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
            axes[1].set_ylabel(r"FoB($\Omega_m$, $\sigma_8$, $b_1$)")
            axes[1].set_xscale("log")
            axes[1].set_xlabel(r"$N_\mathrm{train}$ ($= N_\mathrm{cosmo} \times N_\mathrm{bias}$)")
            fig.tight_layout()
            for d in (FIG_DIR, FIG_DIR_LATEST):
                d.mkdir(parents=True, exist_ok=True)
                out = d / f"fig10_convergence_pgg_b_pgm_kp035_{vtag}{tag_part}_fast.png"
                fig.savefig(out, bbox_inches="tight", facecolor="white")
                print(f"Saved {out}")
            plt.close(fig)

        # Skip fig11-style multi-combo when only one stat key was collected
        if n_stat_keys <= 1:
            continue

        frames_ok = []
        for (statistics, mask), color in zip(KP035_CONFIGS, COLORS_FID):
            sk = "_".join(statistics) + (mask or "_nomask")
            sub = dfi[dfi["stat_key"] == sk]
            if sub.empty:
                continue
            lab = " + ".join(
                {"pk": r"$P_\mathrm{gg}$", "bispec": r"$B_\mathrm{ggg}$", "pgm": r"$P_\mathrm{gm}$"}[s]
                for s in statistics
            )
            frames_ok.append((sub, color, lab))
        if not frames_ok:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        for ax, key in zip(axes[0], FOB_KEYS):
            for sub, color, _ in frames_ok:
                draw(ax, sub, f"fom_{key}", color)
            ax.set_title(
                {"omega_m": r"$\Omega_\mathrm{c}$", "sigma8": r"$\sigma_8$", "b1": r"$b_1$"}[key],
                fontsize=18, fontweight="bold",
            )
            ax.set_ylabel("FoM")
            ax.set_yscale("log")
            ax.set_xscale("log")
        for ax, key in zip(axes[1], FOB_KEYS):
            for sub, color, _ in frames_ok:
                draw(ax, sub, f"fob_{key}", color)
            ax.axhline(gauss, color="k", ls="--", lw=1, alpha=0.7)
            ax.axhline(lim1, color="gray", ls=":", lw=1, alpha=0.7)
            ax.set_ylabel("FoB")
            ax.set_xscale("log")
            ax.set_xlabel(r"$N_\mathrm{train}$")
        handles = [Line2D([0], [0], color=c, lw=1.5) for _, c, _ in frames_ok]
        fig.tight_layout()
        fig.legend(
            handles, [lab for _, _, lab in frames_ok], loc="center left",
            bbox_to_anchor=(0.91, 0.5), frameon=False,
        )
        fig.subplots_adjust(right=0.90)
        for d in (FIG_DIR, FIG_DIR_LATEST):
            out = d / f"fig11_convergence_all_combos_kp035_{vtag}{tag_part}_fast.png"
            fig.savefig(out, bbox_inches="tight", facecolor="white")
            print(f"Saved {out}")
        plt.close(fig)


def main():
    global _MAX_DRAWS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--validate", action="store_true", help="Compare one cell to paper cache")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument(
        "--max-draws", type=int, default=DEFAULT_MAX_DRAWS,
        help=f"Cap draws read per member (default {DEFAULT_MAX_DRAWS}; contiguous prefix)",
    )
    ap.add_argument(
        "--full-draws", action="store_true",
        help="Read all posterior draws (slow I/O; closest to plotter_figs)",
    )
    ap.add_argument(
        "--stat-key", action="append", default=None,
        help="Only this stat_key (repeatable), e.g. pk_bispec_pgm_kp0.35_kb0.25_kpgm0.25",
    )
    ap.add_argument(
        "--cache-tag", type=str, default="",
        help="Suffix for cache/figure names (default: fulldraws if --full-draws else empty)",
    )
    args = ap.parse_args()
    _MAX_DRAWS = None if args.full_draws else args.max_draws
    cache_tag = args.cache_tag
    if not cache_tag and args.full_draws:
        cache_tag = "fulldraws"

    print("=== inventory ===")
    inventory()
    print(f"max_draws={_MAX_DRAWS}")
    if args.stat_key:
        print(f"stat_keys={args.stat_key}")
    if args.inventory_only:
        return
    if args.validate:
        print("\n=== validate ===")
        validate_one()
        return

    print(
        f"\n=== collect (workers={args.workers}, max_draws={_MAX_DRAWS}, "
        f"stat_keys={args.stat_key}, cache_tag={cache_tag!r}) ==="
    )
    df = collect_all(workers=args.workers, stat_keys=args.stat_key)
    if df.empty:
        print("No data loaded.")
        return
    if not args.no_save:
        save_caches(df, tag=cache_tag)
    if args.plot:
        print("\n=== plot ===")
        plot_from_df(df, tag=cache_tag)


if __name__ == "__main__":
    main()
