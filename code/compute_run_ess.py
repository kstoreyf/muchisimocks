#!/usr/bin/env python3
# Run from repo root, e.g.:
#   python code/compute_run_ess.py
#   python code/compute_run_ess.py --noise-modes noiseless noisy --overwrite
#   python code/compute_run_ess.py --stat-labels pk pk_pgm pk_bispec_kb025 --output-dir results/ess_run_scores
"""
Compute direct-posterior min-ESS proxy for every run under each sweep directory,
matching ``2026-04-08_inference_scaling_diagnosis.ipynb`` (cached scalers + batched arviz.ess).

Writes one CSV per run combo, named ``ess{tag_inf}.csv`` where ``tag_inf`` matches
``generate_train_config`` sweep configs (noise mode, statistics, ``tags_mask``, params, sweep). Updates after each run.
With ``--overwrite`` false (default), loads existing CSV and skips runs already marked ok.

Run (from repo root), examples::

    python code/compute_run_ess.py
    python code/compute_run_ess.py --noise-modes noiseless noisy --overwrite
    python code/compute_run_ess.py --stat-labels pk pk_pgm --output-dir /path/to/out
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import data_loader  # noqa: E402
import paths  # noqa: E402
import wandb  # noqa: E402
from generate_config_inference import (  # noqa: E402
    BX_SWEEP,
    N_TRAIN_SWEEP,
    NOISE_MODES,
    resolve_test_scenario_tags,
    resolve_train_tag_bundle,
)
# Match ``2026-04-08_inference_scaling_diagnosis.ipynb`` W&B extras.
WANDB_PROJECT = "muchisimocks-sbi"
EXTRA_CFG_KEYS = ("max_epochs", "validation_fraction", "model_type")

_RUN_DIR_RE = re.compile(r"^[a-z0-9]{8,}$")

# Same layout as ``generate_config_inference.main`` (``stat_arr`` / ``tags_mask_arr``).
STAT_ARR: list[list[str]] = [
    ["pk"],
    ["pk", "pgm"],
    ["pk", "bispec"],
    ["pk", "bispec", "pgm"],
]
TAGS_MASK_ARR: list[list[str]] = [
    [""],
    ["", ""],
    ["", "_kb0.25"],
    ["", "_kb0.25", ""],
]
assert len(TAGS_MASK_ARR) == len(STAT_ARR), "TAGS_MASK_ARR must align with STAT_ARR"
for _i, (stats, masks) in enumerate(zip(STAT_ARR, TAGS_MASK_ARR)):
    assert len(masks) == len(stats), (
        f"row {_i}: len(tags_mask) must match len(statistics): "
        f"statistics={stats!r}, tags_mask={masks!r}"
    )


def stat_label_from_row(statistics: list[str], tags_mask: list[str]) -> str:
    """Stable filename/CLI label from one ``stat_arr`` row and matching ``tags_mask``."""
    base = "_".join(statistics)
    extra = "".join(tags_mask).replace(".", "")
    return base + extra if extra else base


def default_stat_labels() -> list[str]:
    return [
        stat_label_from_row(s, m) for s, m in zip(STAT_ARR, TAGS_MASK_ARR)
    ]


def build_sweep_tag_inf(
    *,
    data_mode: str,
    statistics: list[str],
    tags_mask: list[str],
    tag_params: str,
    tag_biasparams: str,
    tag_noise: str | None,
    reparameterize: bool,
    tag_sweep: str,
) -> str:
    """Same ``tag_inf`` as ``generate_train_config`` with ``run_mode='sweep'`` (see that function)."""
    tag_masks = "".join(tags_mask)
    tag_stats = f'_{"_".join(statistics)}'
    tag_paramsall = tag_params + tag_biasparams
    if tag_noise is not None:
        tag_paramsall += tag_noise
    tag_data = "_" + data_mode + tag_stats + tag_masks + tag_paramsall
    tag_inf_num_sweep = f"_bx{BX_SWEEP}_ntrain{N_TRAIN_SWEEP}"
    base_inf_sweep = tag_data + ("_rp" if reparameterize else "") + tag_inf_num_sweep
    return base_inf_sweep + f"_sweep{tag_sweep}"


def sweep_root_for_tag_inf(tag_inf: str) -> Path:
    return paths.DIR_RESULTS / "results_sbi" / f"sbi{tag_inf}"


def _x_scaled_from_scalers(
    scalers: dict, statistics: list[str], y_unscaled_blocks: list, obs_idx: int
) -> np.ndarray:
    parts = [
        scalers[stat].scale(y_unscaled_blocks[i][obs_idx : obs_idx + 1])
        for i, stat in enumerate(statistics)
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def posterior_draws_min_ess(posterior, x_np: np.ndarray, n_samples: int) -> float:
    import arviz as az
    import torch

    x_t = torch.as_tensor(x_np, dtype=torch.float32)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)
    s = (
        posterior.sample(
            (n_samples,),
            x=x_t,
            show_progress_bars=False,
        )
        .detach()
        .cpu()
        .numpy()
    )
    npar = int(s.shape[-1])
    idata = az.from_dict(posterior={f"p{k}": s[np.newaxis, :, k] for k in range(npar)})
    ess_ds = az.ess(idata)
    return float(
        min(float(np.asarray(ess_ds[v].values).squeeze()) for v in ess_ds.data_vars)
    )


def load_coverage_blocks(
    *,
    data_mode: str,
    statistics: list[str],
    tags_mask: list[str],
    tag_params_cov: str,
    noise_mode: str,
    n_cosmo_max: int | None,
    n_cov_rows: int | None,
) -> list[np.ndarray]:
    cov = resolve_test_scenario_tags("coverage", noise_mode, tag_params_cov)
    tag_bias_cov = cov["tag_biasparams_test"]
    tag_noise_cov = cov["tag_noise_test"]
    _, y_raw, _, *_ = data_loader.load_data(
        data_mode,
        statistics,
        tag_params_cov,
        tag_bias_cov,
        tag_noise=tag_noise_cov,
        tags_mask=tags_mask,
        bx=None,
        n_cosmo_max=n_cosmo_max,
    )
    n_rows = int(y_raw[0].shape[0])
    n_use = n_rows if n_cov_rows is None else min(int(n_cov_rows), n_rows)
    return [np.asarray(y_raw[i][:n_use], dtype=np.float64) for i in range(len(statistics))]


def list_run_dirs(sweep_root: Path) -> list[Path]:
    out = []
    if not sweep_root.is_dir():
        return out
    for p in sorted(sweep_root.iterdir()):
        if (
            p.is_dir()
            and _RUN_DIR_RE.match(p.name) is not None
            and (p / "inference.p").is_file()
        ):
            out.append(p)
    return out


CSV_FIELDNAMES = [
    "noise_mode",
    "stat_label",
    "tag_sweep",
    "run_id",
    "status",
    "min_ess",
    "seconds",
    "error",
    "statistics",
    "tags_mask",
    "tag_inf",
    "n_posterior_draws",
    "n_obs",
    "n_cosmo_max",
    "n_cov_rows",
    "per_obs_ess",
]


def _encode_list_for_csv(xs: list[str]) -> str:
    return ";".join(xs)


def _decode_list_from_csv(s: str) -> list[str]:
    return s.split(";") if s else []


def _encode_per_obs_ess(xs: list[float] | None) -> str:
    if xs is None:
        return ""
    return "|".join(f"{float(v):.8g}" for v in xs)


def read_ess_csv(path: Path) -> dict[str, dict]:
    """Map run_id -> row dict (string values as stored)."""
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rid = row.get("run_id")
            if rid:
                out[rid] = row
    return out


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDNAMES})
    tmp.replace(path)


def csv_row_for_run(
    *,
    noise_mode: str,
    stat_label: str,
    tag_sweep: str,
    run_id: str,
    status: str,
    min_ess: float | None,
    seconds: float | None,
    error: str | None,
    statistics: list[str],
    tags_mask: list[str],
    tag_inf: str,
    n_samples: int,
    n_obs: int,
    n_cosmo_max: int | None,
    n_cov_rows: int | None,
    per_obs_ess: list[float] | None,
) -> dict[str, str]:
    return {
        "noise_mode": noise_mode,
        "stat_label": stat_label,
        "tag_sweep": tag_sweep,
        "run_id": run_id,
        "status": status,
        "min_ess": "" if min_ess is None else f"{min_ess:.8g}",
        "seconds": "" if seconds is None else f"{seconds:.4f}",
        "error": "" if error is None else error,
        "statistics": _encode_list_for_csv(statistics),
        "tags_mask": _encode_list_for_csv(tags_mask),
        "tag_inf": tag_inf,
        "n_posterior_draws": str(n_samples),
        "n_obs": str(n_obs),
        "n_cosmo_max": "" if n_cosmo_max is None else str(n_cosmo_max),
        "n_cov_rows": "" if n_cov_rows is None else str(n_cov_rows),
        "per_obs_ess": _encode_per_obs_ess(per_obs_ess),
    }


def pick_best_ess_run_row(rows: dict[str, dict]) -> dict | None:
    """Row with largest ``min_ess`` among ``status == ok`` (higher is better)."""
    best: dict | None = None
    best_m: float | None = None
    for r in rows.values():
        if r.get("status") != "ok":
            continue
        s = str(r.get("min_ess", "")).strip()
        if not s:
            continue
        try:
            m = float(s)
        except ValueError:
            continue
        if best_m is None or m > best_m:
            best_m = m
            best = r
    return best


def load_losses_from_inference_p(dir_sbi: Path):
    """Training/validation loss series from ``inference._summary`` (notebook-aligned)."""
    fn = dir_sbi / "inference.p"
    if not fn.is_file():
        return None
    with open(fn, "rb") as f:
        inf = pickle.load(f)
    summ = getattr(inf, "_summary", None)
    if not summ:
        return None
    return {
        "training_loss": np.asarray(summ.get("training_loss", []), dtype=float),
        "validation_loss": np.asarray(summ.get("validation_loss", []), dtype=float),
    }


def loss_summary(losses) -> dict:
    if losses is None or losses["training_loss"].size == 0:
        return {}
    tr = losses["training_loss"]
    va = losses["validation_loss"]
    out = {"n_epochs": len(tr), "final_training_loss": float(tr[-1])}
    if va.size:
        out["final_validation_loss"] = float(va[-1])
        out["min_validation_loss"] = float(np.min(va))
        out["min_validation_epoch"] = int(np.argmin(va)) + 1
    return out


def per_obs_ess_from_csv_cell(s: str) -> list[float] | None:
    if not str(s).strip():
        return None
    return [float(x) for x in str(s).split("|")]


def print_scaler_train_bounds(dir_sbi: Path, statistics: list[str]) -> None:
    print("  Scaler train bounds (see notebook: x_train_min / x_train_max)", flush=True)
    for s in statistics:
        p = dir_sbi / f"scaler_y_{s}.p"
        if not p.is_file():
            print(f"    [{s}] missing {p.name!r}", flush=True)
            continue
        with open(p, "rb") as f:
            sa = pickle.load(f)
        amin = np.asarray(sa.x_train_min)
        amax = np.asarray(sa.x_train_max)
        fn = getattr(sa, "func", None)
        print(
            f"    [{s}] func={fn!r}  "
            f"x_train_min range [{float(amin.min()):.4g}, {float(amin.max()):.4g}]  "
            f"x_train_max range [{float(amax.min()):.4g}, {float(amax.max()):.4g}]  "
            f"shape={amin.shape}",
            flush=True,
        )


def print_ess_best_report(*, sweep_root: Path, rows: dict[str, dict], statistics: list[str]) -> None:
    """Print W&B hparams, loss summary, scalers for the highest min-ESS run (notebook-style)."""
    best = pick_best_ess_run_row(rows)
    if best is None:
        print(
            "[ess-best] no successful runs with min_ess — skipping best-run report",
            flush=True,
        )
        return
    run_id = str(best["run_id"])
    run_dir = sweep_root / run_id
    m_best = float(str(best.get("min_ess", "nan")).strip())
    n_obs_s = str(best.get("n_obs", "")).strip()
    n_obs_i = int(n_obs_s) if n_obs_s.isdigit() else None
    per_obs = per_obs_ess_from_csv_cell(str(best.get("per_obs_ess", "")))

    print("=" * 72, flush=True)
    print("ESS-best (direct posterior + arviz.ess proxy)", flush=True)
    print(f"  run_id: {run_id}", flush=True)
    print(f"  min over obs: {m_best:.6g}", flush=True)
    if n_obs_i is not None:
        print(f"  n_obs (coverage rows scored): {n_obs_i}", flush=True)
    if per_obs:
        print(
            f"  per-obs ESS range [{min(per_obs):.4f}, {max(per_obs):.4f}]",
            flush=True,
        )
    print(f"  dir: {run_dir}", flush=True)

    try:
        from sbi_model import _WANDB_SWEEP_PARAMETER_KEYS

        api = wandb.Api()
        wb_run = api.run(f"{api.default_entity}/{WANDB_PROJECT}/{run_id}")
        cfg, sm = dict(wb_run.config), dict(wb_run.summary)
        bvl = sm.get("best_validation_loss")
        print(f"  best_validation_loss (W&B summary): {bvl}", flush=True)
        for k in _WANDB_SWEEP_PARAMETER_KEYS:
            print(f"  {k}: {cfg.get(k)}", flush=True)
        for k in EXTRA_CFG_KEYS:
            if cfg.get(k) is not None:
                print(f"  {k}: {cfg[k]}", flush=True)
    except Exception as ex:
        print(f"  (W&B lookup skipped: {ex})", flush=True)

    losses = load_losses_from_inference_p(run_dir)
    ls = loss_summary(losses)
    if ls:
        print(f"  loss history (inference._summary): {ls}", flush=True)
    else:
        print(
            f"  loss history: none or empty in {run_dir / 'inference.p'}",
            flush=True,
        )

    print_scaler_train_bounds(run_dir, statistics)

    cfg_pkl = run_dir / "config.pkl"
    if cfg_pkl.is_file():
        with open(cfg_pkl, "rb") as f:
            raw = pickle.load(f)
        if raw == {}:
            print(
                "  note: config.pkl is empty dict — fit_model saves vars(wandb.config); "
                "often no public keys, so use W&B above.",
                flush=True,
            )
        elif isinstance(raw, dict) and raw:
            print(f"  config.pkl ({len(raw)} keys):", flush=True)
            for k, v in list(raw.items())[:24]:
                print(f"    {k}: {v!r}", flush=True)
            if len(raw) > 24:
                print(f"    ... ({len(raw) - 24} more keys)", flush=True)


def process_one_run(
    run_dir: Path,
    *,
    posterior,
    scalers: dict,
    statistics: list[str],
    y_cov_nb: list[np.ndarray],
    n_obs: int,
    n_samples: int,
) -> tuple[float, list[float], float]:
    t0 = time.perf_counter()
    ess_scores = []
    for j in range(n_obs):
        x_o = _x_scaled_from_scalers(scalers, statistics, y_cov_nb, obs_idx=j)
        ess_scores.append(posterior_draws_min_ess(posterior, x_o, n_samples))
    dt = time.perf_counter() - t0
    return float(min(ess_scores)), ess_scores, dt


def run_combo(
    *,
    noise_mode: str,
    stat_label: str,
    statistics: list[str],
    tags_mask: list[str],
    args: argparse.Namespace,
) -> None:
    train_kw = resolve_train_tag_bundle(args.tag_params_train, noise_mode)
    tag_inf = build_sweep_tag_inf(
        data_mode=args.data_mode,
        statistics=statistics,
        tags_mask=tags_mask,
        tag_params=train_kw["tag_params"],
        tag_biasparams=train_kw["tag_biasparams"],
        tag_noise=train_kw["tag_noise"],
        reparameterize=args.reparameterize,
        tag_sweep=args.tag_sweep,
    )
    sweep_root = sweep_root_for_tag_inf(tag_inf)
    sweep_slug = args.tag_sweep.lstrip("-") or "sweep"
    out_path = args.output_dir / f"ess_{noise_mode}_{stat_label}_{sweep_slug}.csv"

    rows: dict[str, dict] = {}
    if not args.overwrite and out_path.is_file():
        rows = read_ess_csv(out_path)

    run_dirs = list_run_dirs(sweep_root)
    if not run_dirs:
        print(f"[skip] no runs under {sweep_root}", flush=True)
        atomic_write_csv(out_path, [rows[r] for r in sorted(rows)])
        return

    y_cov_nb = load_coverage_blocks(
        data_mode=args.data_mode,
        statistics=statistics,
        tags_mask=tags_mask,
        tag_params_cov=args.tag_params_cov,
        noise_mode=noise_mode,
        n_cosmo_max=args.n_cosmo_max,
        n_cov_rows=args.n_cov_rows,
    )
    n_rows = int(y_cov_nb[0].shape[0])
    n_obs = n_rows if args.n_obs is None else min(int(args.n_obs), n_rows)

    print(
        f"=== {noise_mode} | {stat_label} | {len(run_dirs)} run dir(s) | "
        f"{n_obs} obs | {sweep_root.name}",
        flush=True,
    )

    for rd in run_dirs:
        rid = rd.name
        prev = rows.get(rid)
        if (
            not args.overwrite
            and isinstance(prev, dict)
            and prev.get("status") == "ok"
            and str(prev.get("min_ess", "")).strip() != ""
        ):
            print(f"  [resume] skip ok {rid}", flush=True)
            continue

        try:
            with open(rd / "inference.p", "rb") as f:
                inf = pickle.load(f)
            posterior = inf.build_posterior(sample_with="direct")
            scalers = {}
            for stat in statistics:
                with open(rd / f"scaler_y_{stat}.p", "rb") as f:
                    scalers[stat] = pickle.load(f)
            min_ess, per_obs, dt = process_one_run(
                rd,
                posterior=posterior,
                scalers=scalers,
                statistics=statistics,
                y_cov_nb=y_cov_nb,
                n_obs=n_obs,
                n_samples=args.n_samples,
            )
            rows[rid] = csv_row_for_run(
                noise_mode=noise_mode,
                stat_label=stat_label,
                tag_sweep=args.tag_sweep,
                run_id=rid,
                status="ok",
                min_ess=min_ess,
                seconds=dt,
                error=None,
                statistics=statistics,
                tags_mask=tags_mask,
                tag_inf=tag_inf,
                n_samples=args.n_samples,
                n_obs=n_obs,
                n_cosmo_max=args.n_cosmo_max,
                n_cov_rows=args.n_cov_rows,
                per_obs_ess=per_obs,
            )
            print(
                f"  [ok] {rid} min_ess={min_ess:.6g} dt={dt:.2f}s",
                flush=True,
            )
        except Exception as ex:
            rows[rid] = csv_row_for_run(
                noise_mode=noise_mode,
                stat_label=stat_label,
                tag_sweep=args.tag_sweep,
                run_id=rid,
                status="error",
                min_ess=None,
                seconds=None,
                error=repr(ex),
                statistics=statistics,
                tags_mask=tags_mask,
                tag_inf=tag_inf,
                n_samples=args.n_samples,
                n_obs=n_obs,
                n_cosmo_max=args.n_cosmo_max,
                n_cov_rows=args.n_cov_rows,
                per_obs_ess=None,
            )
            print(f"  [error] {rid}: {ex}", flush=True)

        atomic_write_csv(out_path, [rows[r] for r in sorted(rows)])

    print(f"  wrote {out_path}", flush=True)
    print_ess_best_report(sweep_root=sweep_root, rows=rows, statistics=statistics)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=paths.DIR_RESULTS / "results_sbi/run_stats_ess",
        help="Directory for ess{tag_inf}.csv (tag_inf matches training sweep naming; see generate_config_inference)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing CSV and recompute all runs",
    )
    p.add_argument("--data-mode", default="muchisimocks")
    p.add_argument("--tag-params-train", default="_p5_n10000")
    p.add_argument("--tag-params-cov", default="_coverage_p5_n1000")
    p.add_argument("--tag-sweep", default="-rand30", help="Sweep suffix, e.g. -rand30")
    p.add_argument(
        "--reparameterize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--n-samples", type=int, default=10, help="Posterior draws per obs")
    p.add_argument(
        "--n-obs",
        type=int,
        default=5,
        help="Max coverage rows to score (default: all loaded)",
    )
    p.add_argument(
        "--n-cosmo-max",
        type=int,
        default=1000,
        help="n_cosmo_max passed to load_data for coverage",
    )
    p.add_argument(
        "--n-cov-rows",
        type=int,
        default=None,
        help="Cap rows after load (default: all rows returned)",
    )
    p.add_argument(
        "--noise-modes",
        nargs="*",
        default=list(NOISE_MODES),
        choices=list(NOISE_MODES),
        help="Subset of noiseless noisy",
    )
    p.add_argument(
        "--stat-labels",
        nargs="*",
        default=default_stat_labels(),
        help="Subset of labels from stat_arr/tags_mask_arr (see STAT_ARR in script)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    label_to_opt = {
        stat_label_from_row(s, m): (s, m)
        for s, m in zip(STAT_ARR, TAGS_MASK_ARR)
    }
    for lab in args.stat_labels:
        if lab not in label_to_opt:
            raise SystemExit(
                f"Unknown stat label {lab!r}. Choose from {list(label_to_opt)}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for noise_mode in args.noise_modes:
        for stat_label in args.stat_labels:
            statistics, tags_mask = label_to_opt[stat_label]
            run_combo(
                noise_mode=noise_mode,
                stat_label=stat_label,
                statistics=statistics,
                tags_mask=tags_mask,
                args=args,
            )


if __name__ == "__main__":
    main()
