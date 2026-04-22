#!/usr/bin/env python3
"""
Fix / backfill W&B sweep runs and local result dirs.

Two modes (use one at a time):

1. **Default — backfill ``best_validation_loss``** (historical sweeps)

   For each finished run in the named sweep, reads local
   ``<DIR_RESULTS>/results_sbi/sbi<sweep_name>/<run_id>/inference.p``,
   takes ``min(validation_loss)``, and writes that value to the run's W&B summary.

2. **`--write-config-pkl` — local ``config.pkl`` from W&B run config**

   For each finished run, fetches hyperparameters via the Public API
   (same keys as ``dict(run.config)`` without leading ``_``) and writes
   ``<run_dir>/config.pkl``. Use for trials where training saved an empty
   ``config.pkl`` (old ``vars(wandb.config)`` bug) or missing file.

W&B does not allow changing a sweep's objective metric after the sweep has started;
this script does not update sweep config.

Training code (``sbi_model.fit_model``) logs ``best_validation_loss`` for new runs.
Use mode (1) for historical sweeps trained before that change.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import wandb

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import paths  # noqa: E402


def dir_sbi_from_sweep_name(sweep_name: str) -> pathlib.Path:
    """
    Match ``SBIModel`` output dir: ``DIR_RESULTS/results_sbi/sbi{tag_sbi}``.

    For sweep-mode training configs, ``tag_inf`` and W&B ``sweep_name`` are the same
    string, so sweep_name alone determines the results path.
    """
    return paths.DIR_RESULTS / "results_sbi" / f"sbi{sweep_name}"


def wandb_run_config_to_plain_dict(run) -> dict:
    """Match ``sbi_model._training_config_to_plain_dict`` for W&B run configs."""
    return {
        k: v
        for k, v in dict(run.config).items()
        if not str(k).startswith("_")
    }


def _config_pkl_nonempty(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return True
    return isinstance(data, dict) and len(data) > 0


def cmd_backfill_loss(args: argparse.Namespace) -> int:
    root = (
        args.dir_sbi.resolve()
        if args.dir_sbi is not None
        else dir_sbi_from_sweep_name(args.sweep_name)
    )

    wandb.login()
    api = wandb.Api()
    entity = args.entity or api.default_entity
    proj = args.project

    sweeps = api.project(proj, entity=entity).sweeps()
    sweep = next((s for s in sweeps if s.name == args.sweep_name), None)
    if sweep is None:
        print(f"ERROR: no sweep named {args.sweep_name!r} in {entity}/{proj}", file=sys.stderr)
        return 1

    print(f"Mode: backfill best_validation_loss to W&B summary", flush=True)
    print(f"Using dir_sbi={root}", flush=True)
    if not root.is_dir():
        print(f"ERROR: dir_sbi is not a directory: {root}", file=sys.stderr)
        return 1

    n_ok = 0
    n_skip = 0
    n_missing = 0

    for run in sweep.runs:
        state = str(getattr(run, "state", "")).lower()
        if state not in {"finished", "completed"}:
            n_skip += 1
            continue

        run_dir = root / run.id
        inference_path = run_dir / "inference.p"
        if not inference_path.is_file():
            print(f"[{run.id}] inference.p not found under {run_dir}, skipping")
            n_missing += 1
            continue

        with open(inference_path, "rb") as f:
            inference = pickle.load(f)

        train_log = getattr(inference, "_summary", None) or {}
        val_losses = train_log.get("validation_loss") or []
        if not val_losses:
            print(f"[{run.id}] no validation_loss series in inference._summary, skipping")
            n_missing += 1
            continue

        best_val_loss = float(min(val_losses))
        print(f"[{run.id}] best_validation_loss = {best_val_loss:.6g}")

        if args.dry_run:
            n_ok += 1
            continue

        api_run = api.run(f"{entity}/{proj}/{run.id}")
        api_run.summary["best_validation_loss"] = best_val_loss
        api_run.update()
        n_ok += 1

    print(
        f"Done: updated={n_ok} missing_or_empty={n_missing} skipped_nonterminal={n_skip} dry_run={args.dry_run}",
        flush=True,
    )
    return 0


def cmd_write_config_pkl(args: argparse.Namespace) -> int:
    root = (
        args.dir_sbi.resolve()
        if args.dir_sbi is not None
        else dir_sbi_from_sweep_name(args.sweep_name)
    )

    wandb.login()
    api = wandb.Api()
    entity = args.entity or api.default_entity
    proj = args.project

    sweeps = api.project(proj, entity=entity).sweeps()
    sweep = next((s for s in sweeps if s.name == args.sweep_name), None)
    if sweep is None:
        print(f"ERROR: no sweep named {args.sweep_name!r} in {entity}/{proj}", file=sys.stderr)
        return 1

    print(f"Mode: write local config.pkl from W&B run.config", flush=True)
    print(f"Using dir_sbi={root}", flush=True)
    if not root.is_dir():
        print(f"ERROR: dir_sbi is not a directory: {root}", file=sys.stderr)
        return 1

    n_ok = 0
    n_skip = 0
    n_skip_nonempty = 0
    n_missing_dir = 0
    n_empty_config = 0

    for run in sweep.runs:
        state = str(getattr(run, "state", "")).lower()
        if state not in {"finished", "completed"}:
            n_skip += 1
            continue

        run_dir = root / run.id
        if not run_dir.is_dir():
            print(f"[{run.id}] local run dir missing {run_dir}, skipping")
            n_missing_dir += 1
            continue

        cfg_path = run_dir / "config.pkl"
        if not args.force and _config_pkl_nonempty(cfg_path):
            print(f"[{run.id}] config.pkl already non-empty, skip (use --force to overwrite)")
            n_skip_nonempty += 1
            continue

        plain = wandb_run_config_to_plain_dict(run)
        if not plain:
            print(f"[{run.id}] W&B run.config is empty after filtering; skipping")
            n_empty_config += 1
            continue

        print(f"[{run.id}] writing config.pkl ({len(plain)} keys): {sorted(plain.keys())}")

        if args.dry_run:
            n_ok += 1
            continue

        with open(cfg_path, "wb") as f:
            pickle.dump(plain, f)
        n_ok += 1

    print(
        f"Done: written={n_ok} skip_nonempty={n_skip_nonempty} missing_dir={n_missing_dir} "
        f"empty_wandb_config={n_empty_config} skipped_nonterminal={n_skip} dry_run={args.dry_run}",
        flush=True,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Fix W&B sweep metrics or local config.pkl. "
            "Default: backfill best_validation_loss from inference.p. "
            "Use --write-config-pkl to pull run.config into each run's config.pkl."
        )
    )
    p.add_argument(
        "sweep_name",
        type=str,
        help="W&B sweep name (same as config sweep_name / tag_inf for sweep training)",
    )
    p.add_argument(
        "--write-config-pkl",
        action="store_true",
        help=(
            "Instead of backfilling W&B summary: fetch each run's config from the API "
            "and write <dir_sbi>/<run_id>/config.pkl (skips non-empty files unless --force)."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Only used with --write-config-pkl: overwrite existing non-empty config.pkl",
    )
    p.add_argument(
        "--dir-sbi",
        type=pathlib.Path,
        default=None,
        help="Override results directory (default: DIR_RESULTS/results_sbi/sbi<sweep_name>)",
    )
    p.add_argument(
        "--project",
        default="muchisimocks-sbi",
        help="W&B project name (default: muchisimocks-sbi)",
    )
    p.add_argument(
        "--entity",
        default=None,
        help="W&B entity (default: wandb.Api().default_entity)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not write to W&B or disk",
    )
    args = p.parse_args()

    if args.write_config_pkl:
        return cmd_write_config_pkl(args)
    return cmd_backfill_loss(args)


if __name__ == "__main__":
    raise SystemExit(main())
