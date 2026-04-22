#!/usr/bin/env python3
"""
Rank sweep runs by min validation loss, smoke-test in order, write ``best_run_log.csv``
and ``best_run.txt`` under the sweep ``dir_sbi``. Stops after the first run that passes.

Use ``--tags_stat`` like ``_pk_bispec`` (leading ``_`` optional). Masking is chosen
automatically: fiducial empty ``''`` per statistic, except ``_kb0.25`` on ``bispec``
when that statistic is present.

Use ``--overwrite`` to discard resume data in ``best_run_log.csv`` and re-run stall tests.

Requires PyTorch and sbi. Run from repo root: ``python code/choose_best_run.py``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

try:
    import torch  # noqa: F401
    import sbi  # noqa: F401
except ImportError as e:
    sys.stderr.write(
        "choose_best_run.py requires PyTorch and sbi: %s\n" % (e,)
    )
    raise SystemExit(1) from e

import data_loader  # noqa: E402
import paths  # noqa: E402
import sbi_model  # noqa: E402
from generate_config_inference import (  # noqa: E402
    BX_SWEEP,
    N_TRAIN_SWEEP,
    NOISE_MODES,
    resolve_test_scenario_tags,
    resolve_train_tag_bundle,
    tags_mask_for_sweep,
)

_RUN_DIR_RE = re.compile(r"^[a-z0-9]{8,}$")

BEST_RUN_LOG_CSV = "best_run_log.csv"
BEST_RUN_ID_TXT = "best_run.txt"


def parse_tags_stat(tags_stat: str) -> tuple[list[str], list[str]]:
    """Parse ``_pk_bispec`` → statistics; sweep masks via ``tags_mask_for_sweep`` (``TAG_MASK_BISPEC_SWEEP`` on bispec)."""
    s = tags_stat.strip()
    if not s:
        raise ValueError("tags_stat is empty")
    s = s.lstrip("_")
    if not s:
        raise ValueError("tags_stat has no statistics")
    statistics = [p for p in s.split("_") if p]
    if not statistics:
        raise ValueError("no statistics in tags_stat (e.g. _pk_bispec)")
    return statistics, tags_mask_for_sweep(statistics)


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


def list_run_dirs(sweep_root: Path) -> list[Path]:
    if not sweep_root.is_dir():
        return []
    out = []
    for p in sorted(sweep_root.iterdir()):
        if (
            p.is_dir()
            and _RUN_DIR_RE.match(p.name) is not None
            and (p / "inference.p").is_file()
        ):
            out.append(p)
    return out


def load_losses_from_inference_p(dir_sbi: Path):
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
    _, y_raw, _, *_ = data_loader.load_data(
        data_mode,
        statistics,
        tag_params_cov,
        cov["tag_biasparams_test"],
        tag_noise=cov["tag_noise_test"],
        tags_mask=tags_mask,
        bx=None,
        n_cosmo_max=n_cosmo_max,
    )
    n_rows = int(y_raw[0].shape[0])
    n_use = n_rows if n_cov_rows is None else min(int(n_cov_rows), n_rows)
    return [np.asarray(y_raw[i][:n_use], dtype=np.float64) for i in range(len(statistics))]


def _tag_sbi_for_run_dir(sweep_root: Path, run_dir: Path) -> str:
    name = sweep_root.name
    if not name.startswith("sbi") or len(name) < 4:
        raise ValueError("Unexpected sweep_root.name: %r" % (name,))
    return "%s/%s" % (name[3:], run_dir.name)


def _rank_run_dirs(
    run_dirs: list[Path],
) -> list[tuple[Path, float | None, dict]]:
    scored = []
    for rd in run_dirs:
        losses = load_losses_from_inference_p(rd)
        ls = loss_summary(losses) if losses else {}
        m = ls.get("min_validation_loss")
        m_float = float(m) if m is not None else None
        scored.append((rd, m_float, ls))
    scored.sort(
        key=lambda t: (
            t[1] is None,
            t[1] if t[1] is not None else float("inf"),
            t[0].name,
        )
    )
    return scored


# --- best_run_log.csv: resume + full ranking each write --------------------

def read_best_run_log(path: Path) -> dict[str, dict[str, str]]:
    """run_name -> {best_val_loss, pass_stall_test}; pass empty if pending."""
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        r = csv.DictReader(f, dialect=dialect)
        for row in r:
            rid = (row.get("run_name") or "").strip()
            if not rid:
                continue
            out[rid] = {
                "best_val_loss": (row.get("best_val_loss") or "").strip(),
                "pass_stall_test": (row.get("pass_stall_test") or "").strip().lower(),
            }
    return out


def write_best_run_log(
    path: Path,
    rows: list[tuple[str, float | None, str]],
) -> None:
    """
    rows: (run_name, best_val_loss, pass_stall_test) — pass is '', 'true', or 'false'.
    Sorted by loss (min first) already expected.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["run_name", "best_val_loss", "pass_stall_test"])
        for name, loss, p in rows:
            loss_s = "" if loss is None else "%.8g" % loss
            w.writerow([name, loss_s, p])
    tmp.replace(path)


def write_best_run_id(path: Path, run_id: str) -> None:
    """Single line: wandb run id only (easy ``cat best_run.txt``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(run_id.strip() + "\n", encoding="utf-8")
    tmp.replace(path)


def clear_best_run_id(path: Path) -> None:
    if path.is_file():
        path.unlink()


def merge_rows_for_write(
    ranked: list[tuple[Path, float | None, dict]],
    saved: dict[str, dict[str, str]],
) -> list[tuple[str, float | None, str]]:
    """Fresh val losses; stall pass from file when already 'true' or 'false'."""
    rows: list[tuple[str, float | None, str]] = []
    for rd, mv, _ls in ranked:
        rid = rd.name
        rec = saved.get(rid, {})
        p = rec.get("pass_stall_test", "")
        if p not in ("true", "false"):
            p = ""
        rows.append((rid, mv, p))
    return rows


class _RejectionStallWarningHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.saw_stall_warning = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record).lower()
        except Exception:
            msg = str(record.msg).lower()
        if "proposal samples" in msg and "accepted" in msg:
            self.saw_stall_warning = True


def _coverage_batch_unscaled(
    y_cov_nb: list[np.ndarray],
    start: int,
    batch: int,
) -> list[np.ndarray]:
    out = []
    for block in y_cov_nb:
        end = min(start + batch, block.shape[0])
        out.append(np.asarray(block[start:end], dtype=np.float64))
    return out


def _eval_batch_worker(
    q: Any,
    tag_sbi: str,
    statistics: list[str],
    param_names: list[str],
    y_batch: list[np.ndarray],
    n_samples: int,
) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    h = _RejectionStallWarningHandler()
    root = logging.getLogger()
    root.addHandler(h)
    prev_level = root.level
    root.setLevel(logging.INFO)
    try:
        model = sbi_model.SBIModel(
            tag_sbi=tag_sbi,
            run_mode="load",
            param_names=np.asarray(param_names, dtype=str),
            statistics=statistics,
            overwrite=False,
        )
        model.run()
        t0 = time.perf_counter()
        model.evaluate(y_batch, n_samples=n_samples)
        dt = time.perf_counter() - t0
        q.put(
            (
                "ok",
                {
                    "ok": True,
                    "seconds": float(dt),
                    "stall_warning": bool(h.saw_stall_warning),
                },
            )
        )
    except Exception as ex:
        q.put(("error", {"ok": False, "error": repr(ex)}))
    finally:
        root.removeHandler(h)
        root.setLevel(prev_level)


def run_smoke_eval(
    sweep_root: Path,
    run_dir: Path,
    statistics: list[str],
    y_cov_nb: list[np.ndarray],
    stall_batch_size: int,
    batch_timeout_seconds: float,
    n_samples: int,
) -> dict[str, Any]:
    n_rows = int(y_cov_nb[0].shape[0])
    bs = min(stall_batch_size, n_rows)
    y_batch = _coverage_batch_unscaled(y_cov_nb, 0, bs)
    tag_sbi = _tag_sbi_for_run_dir(sweep_root, run_dir)
    param_names = list(np.loadtxt(run_dir / "param_names.txt", dtype=str))

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_eval_batch_worker,
        args=(q, tag_sbi, statistics, param_names, y_batch, n_samples),
    )
    proc.start()
    try:
        proc.join(timeout=batch_timeout_seconds)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10.0)
            return {
                "status": "timeout",
                "reason": "batch of %d obs exceeded %.0fs" % (bs, batch_timeout_seconds),
            }
        if proc.exitcode not in (0, None):
            return {"status": "error", "reason": "worker exit %s" % proc.exitcode}
        try:
            kind, payload = q.get_nowait()
        except Exception:
            return {"status": "error", "reason": "no result from worker queue"}
        if kind == "error":
            return {"status": "error", **payload}
        if kind == "ok":
            pl = payload
            if pl.get("stall_warning"):
                return {
                    "status": "stall_warning",
                    "reason": "low proposal acceptance",
                    **pl,
                }
            return {"status": "ok", **pl}
        return {"status": "error", "reason": "unknown %r" % (kind,)}
    finally:
        # Avoid resource_tracker "leaked semaphore" warnings when the worker is
        # terminated or when the queue is not fully torn down by the child.
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-mode", default="muchisimocks")
    p.add_argument("--tag-params-train", default="_p5_n10000")
    p.add_argument("--tag-params-cov", default="_coverage_p5_n1000")
    p.add_argument("--tag-sweep", default="-rand30")
    p.add_argument(
        "--reparameterize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    # if stall_batch_size is only 10, won't fail when we know it does later
    p.add_argument("--stall-batch-size", type=int, default=100)
    p.add_argument("--batch-timeout-seconds", type=float, default=300.0)
    # fiducial is 10000. for 1000 with stall_batch_size=100, takes more an 300 seconds for working ones to work
    # aiming for 100 samples seems to be enough
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--n-cosmo-max", type=int, default=1000)
    p.add_argument("--n-cov-rows", type=int, default=None)
    p.add_argument(
        "--noise-modes",
        nargs="*",
        default=list(NOISE_MODES),
        choices=list(NOISE_MODES),
    )
    p.add_argument(
        "--tags_stat",
        required=True,
        metavar="TAGS",
        help="Statistics tag, e.g. _pk_bispec or _pk (masks set automatically; bispec → _kb0.25)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Ignore best_run_log.csv resume state and re-run stall tests from scratch",
    )
    return p.parse_args()


def process_one_combo(
    args: argparse.Namespace,
    noise_mode: str,
    statistics: list[str],
    tags_mask: list[str],
    tags_stat_display: str,
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
    log_path = sweep_root / BEST_RUN_LOG_CSV
    id_path = sweep_root / BEST_RUN_ID_TXT
    run_dirs = list_run_dirs(sweep_root)
    ranked = _rank_run_dirs(run_dirs)

    print("=" * 72)
    print("%s | %s | %s" % (noise_mode, tags_stat_display, sweep_root), flush=True)

    if not ranked:
        print("  No runs with inference.p — skip.", flush=True)
        return

    if args.overwrite:
        saved = {}
        clear_best_run_id(id_path)
        print("  overwrite: ignoring resume; cleared %s" % id_path, flush=True)
    else:
        saved = read_best_run_log(log_path)
        cur_ids = {rd.name for rd, _, _ in ranked}
        saved = {k: v for k, v in saved.items() if k in cur_ids}

    rows = merge_rows_for_write(ranked, saved)
    write_best_run_log(log_path, rows)
    print("  Wrote baseline %s (%d runs)" % (log_path, len(rows)), flush=True)

    y_cov_nb = load_coverage_blocks(
        data_mode=args.data_mode,
        statistics=statistics,
        tags_mask=tags_mask,
        tag_params_cov=args.tag_params_cov,
        noise_mode=noise_mode,
        n_cosmo_max=args.n_cosmo_max,
        n_cov_rows=args.n_cov_rows,
    )

    for i, (rd, mv, _lm) in enumerate(ranked):
        rid = rd.name
        idx = next(j for j, x in enumerate(rows) if x[0] == rid)
        _, _, pcur = rows[idx]
        if pcur in ("true", "false"):
            print(
                "  [%d] %s  loss=%s  pass_stall_test=%s (resume, skip)"
                % (i + 1, rid, mv, pcur),
                flush=True,
            )
            if pcur == "true":
                write_best_run_id(id_path, rid)
                print("  CHOSEN (from file): %s  wrote %s" % (rid, id_path), flush=True)
                return
            continue

        print("  [%d] smoke test %s ..." % (i + 1, rid), flush=True)
        res = run_smoke_eval(
            sweep_root,
            rd,
            statistics,
            y_cov_nb,
            args.stall_batch_size,
            args.batch_timeout_seconds,
            args.n_samples,
        )
        passed = res.get("status") == "ok"
        pnew = "true" if passed else "false"
        rows[idx] = (rid, mv, pnew)
        write_best_run_log(log_path, rows)
        print("    -> %s  wrote %s" % (res, log_path), flush=True)

        if passed:
            write_best_run_id(id_path, rid)
            print(
                "  CHOSEN: %s  min_val_loss=%s  wrote %s"
                % (rid, mv, id_path),
                flush=True,
            )
            return

    clear_best_run_id(id_path)
    print("  No run passed stall test. (cleared %s if present)" % id_path, flush=True)


def main() -> int:
    args = parse_args()
    try:
        statistics, tags_mask = parse_tags_stat(args.tags_stat)
    except ValueError as e:
        sys.stderr.write("tags_stat: %s\n" % e)
        return 1
    tags_stat_display = "_" + "_".join(statistics)

    for noise_mode in args.noise_modes:
        process_one_combo(
            args, noise_mode, statistics, tags_mask, tags_stat_display
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
