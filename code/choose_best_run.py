#!/usr/bin/env python3
"""
Rank sweep runs by min validation loss, smoke-test in order, write ``best_run_log.csv``,
``best_run.txt`` (lowest-loss passing run), and ``best_runs.txt`` (up to ``--n-passing``
passing runs, in loss order). Keeps testing until that many runs pass or the ranked list
is exhausted.

Each candidate run must pass **both** smoke tests (same ``--batch-timeout-seconds``):

1. **Coverage** — first ``--stall-batch-size`` rows; ``--n-samples`` (default 20).
2. **SHAMe OOD** — single mock at ``--shame-tag-mock`` (default ``_nbar0.00022``); ``--n-samples-shame`` (default 100).
   Pass ``--n-samples-shame 0`` to skip SHAMe entirely (coverage-only stall test).

``pass_stall_test`` in the log is ``true`` only when all enabled component columns are ``true``.

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
BEST_RUNS_TXT = "best_runs.txt"

# best_run_log.csv columns (pass_* are '', 'true', or 'false')
LOG_COL_PASS_COVERAGE = "pass_coverage_stall_test"
LOG_COL_PASS_SHAME = "pass_shame_ood_stall_test"
LOG_COL_PASS_COMBINED = "pass_stall_test"
LOG_CSV_HEADER = [
    "run_name",
    "best_val_loss",
    LOG_COL_PASS_COVERAGE,
    LOG_COL_PASS_SHAME,
    LOG_COL_PASS_COMBINED,
]


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


def load_shame_ood_y_obs(
    statistics: list[str],
    tags_mask: list[str],
    tag_mock: str,
) -> list[np.ndarray]:
    """SHAMe OOD mock: one 1D summary vector per statistic (same masks as training)."""
    _, y, _ = data_loader.load_data_ood("shame", statistics, tag_mock, tags_mask=tags_mask)
    return [np.asarray(y[i], dtype=np.float64).ravel() for i in range(len(statistics))]


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

def _norm_pass_cell(value: str | None) -> str:
    p = (value or "").strip().lower()
    return p if p in ("true", "false") else ""


def read_best_run_log(path: Path) -> dict[str, dict[str, str]]:
    """run_name -> log columns; pass fields empty if pending."""
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
                LOG_COL_PASS_COVERAGE: _norm_pass_cell(
                    row.get(LOG_COL_PASS_COVERAGE)
                ),
                LOG_COL_PASS_SHAME: _norm_pass_cell(row.get(LOG_COL_PASS_SHAME)),
                LOG_COL_PASS_COMBINED: _norm_pass_cell(
                    row.get(LOG_COL_PASS_COMBINED) or row.get("pass_stall_test")
                ),
            }
    return out


def write_best_run_log(path: Path, rows: list[tuple[str, float | None, str, str, str]]) -> None:
    """
    rows: (run_name, best_val_loss, pass_coverage, pass_shame, pass_combined).
    Sorted by loss (min first) already expected.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(LOG_CSV_HEADER)
        for name, loss, p_cov, p_shame, p_all in rows:
            loss_s = "" if loss is None else "%.8g" % loss
            w.writerow([name, loss_s, p_cov, p_shame, p_all])
    tmp.replace(path)


def log_row_fully_scored(rec: dict[str, str], *, skip_shame: bool = False) -> bool:
    """True when enabled component smoke tests and combined pass are recorded."""
    cols = [LOG_COL_PASS_COVERAGE, LOG_COL_PASS_COMBINED]
    if not skip_shame:
        cols.insert(1, LOG_COL_PASS_SHAME)
    return all(rec.get(col) in ("true", "false") for col in cols)


def write_best_run_id(path: Path, run_id: str) -> None:
    """Single line: wandb run id only (easy ``cat best_run.txt``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(run_id.strip() + "\n", encoding="utf-8")
    tmp.replace(path)


def clear_best_run_id(path: Path) -> None:
    if path.is_file():
        path.unlink()


def clear_best_runs_ids(path: Path) -> None:
    if path.is_file():
        path.unlink()


def count_passing_rows(rows: list[tuple[str, float | None, str, str, str]]) -> int:
    return sum(1 for _name, _loss, _pc, _ps, p in rows if p == "true")


def passing_run_ids_in_rank_order(
    ranked: list[tuple[Path, float | None, dict]],
    rows: list[tuple[str, float | None, str, str, str]],
) -> list[str]:
    """Run ids with ``pass_stall_test=true``, best validation loss first."""
    pmap = {name: p_all for name, _loss, _pc, _ps, p_all in rows}
    return [rd.name for rd, _mv, _lm in ranked if pmap.get(rd.name) == "true"]


def write_best_run_outputs(
    id_path: Path,
    runs_path: Path,
    ranked: list[tuple[Path, float | None, dict]],
    rows: list[tuple[str, float | None, str, str, str]],
    n_passing: int,
) -> list[str]:
    """
    ``best_run.txt``: lowest-loss passing id (for run_mode=best).
    ``best_runs.txt``: up to ``n_passing`` passing ids, loss order.
    """
    passed = passing_run_ids_in_rank_order(ranked, rows)
    if not passed:
        clear_best_run_id(id_path)
        clear_best_runs_ids(runs_path)
        return []
    write_best_run_id(id_path, passed[0])
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = runs_path.with_name(runs_path.name + ".tmp")
    tmp.write_text("\n".join(passed[:n_passing]) + "\n", encoding="utf-8")
    tmp.replace(runs_path)
    return passed[:n_passing]


def merge_rows_for_write(
    ranked: list[tuple[Path, float | None, dict]],
    saved: dict[str, dict[str, str]],
    *,
    skip_shame: bool = False,
) -> list[tuple[str, float | None, str, str, str]]:
    """Fresh val losses; pass columns from file when fully scored."""
    rows: list[tuple[str, float | None, str, str, str]] = []
    for rd, mv, _ls in ranked:
        rid = rd.name
        rec = saved.get(rid, {})
        if log_row_fully_scored(rec, skip_shame=skip_shame):
            rows.append(
                (
                    rid,
                    mv,
                    rec[LOG_COL_PASS_COVERAGE],
                    rec[LOG_COL_PASS_SHAME],
                    rec[LOG_COL_PASS_COMBINED],
                )
            )
        else:
            rows.append((rid, mv, "", "", ""))
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


def _build_y_batch(
    y_per_stat: list[np.ndarray],
    *,
    start: int,
    batch_size: int,
) -> tuple[list[np.ndarray], int, str]:
    """Return (y_batch for evaluate, n_obs, timeout label fragment)."""
    out = []
    n_obs = 1
    for block in y_per_stat:
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 1:
            out.append(block)
        elif block.ndim == 2:
            end = min(start + batch_size, block.shape[0])
            n_obs = end - start
            out.append(block[start:end])
        else:
            raise ValueError("Unexpected y block shape %s" % (block.shape,))
    if out[0].ndim == 1:
        label = "SHAMe OOD mock (1 obs)"
    else:
        label = "batch of %d coverage obs" % n_obs
    return out, n_obs, label


def run_smoke_eval(
    sweep_root: Path,
    run_dir: Path,
    statistics: list[str],
    y_per_stat: list[np.ndarray],
    batch_timeout_seconds: float,
    n_samples: int,
    *,
    start: int = 0,
    batch_size: int = 1,
) -> dict[str, Any]:
    y_batch, _n_obs, obs_label = _build_y_batch(
        y_per_stat, start=start, batch_size=batch_size
    )
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
                "reason": "%s exceeded %.0fs" % (obs_label, batch_timeout_seconds),
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
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass


def _pass_cell_from_status(status: str) -> str:
    return "true" if status == "ok" else "false"


def run_dual_smoke_eval(
    sweep_root: Path,
    run_dir: Path,
    statistics: list[str],
    y_cov_nb: list[np.ndarray],
    y_shame_ood: list[np.ndarray],
    stall_batch_size: int,
    batch_timeout_seconds: float,
    n_samples_coverage: int,
    n_samples_shame: int,
) -> dict[str, Any]:
    """Coverage batch smoke test, then SHAMe OOD mock; same timeout, sample counts may differ."""
    res_cov = run_smoke_eval(
        sweep_root,
        run_dir,
        statistics,
        y_cov_nb,
        batch_timeout_seconds,
        n_samples_coverage,
        start=0,
        batch_size=stall_batch_size,
    )
    ok_cov = res_cov.get("status") == "ok"
    if n_samples_shame <= 0:
        combined = "ok" if ok_cov else "fail"
        return {
            "status": combined,
            "coverage": res_cov,
            "shame": {"status": "skipped"},
            "pass_coverage": _pass_cell_from_status(res_cov.get("status", "")),
            "pass_shame": "",
        }
    res_shame = run_smoke_eval(
        sweep_root,
        run_dir,
        statistics,
        y_shame_ood,
        batch_timeout_seconds,
        n_samples_shame,
        start=0,
        batch_size=1,
    )
    ok_shame = res_shame.get("status") == "ok"
    combined = "ok" if ok_cov and ok_shame else "fail"
    return {
        "status": combined,
        "coverage": res_cov,
        "shame": res_shame,
        "pass_coverage": _pass_cell_from_status(res_cov.get("status", "")),
        "pass_shame": _pass_cell_from_status(res_shame.get("status", "")),
    }


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
    
    # updated these settings - wasn't getting consistent results with the first setting unfortunately
    #p.add_argument("--batch-timeout-seconds", type=float, default=300.0)
    p.add_argument("--batch-timeout-seconds", type=float, default=300.0)
    # fiducial is 10000. for 1000 with stall_batch_size=100, takes more an 300 seconds for working ones to work
    # aiming for 100 samples seems to be enough
    #p.add_argument("--n-samples", type=int, default=100)
    p.add_argument(
        "--n-samples",
        type=int,
        default=30,
        help="Posterior samples per obs in the coverage stall batch (default 30)",
    )
    p.add_argument(
        "--n-samples-shame",
        type=int,
        default=0,
        help="Posterior samples for the SHAMe OOD stall test (default 1000; 0 = skip SHAMe)",
    )
    p.add_argument(
        "--n-passing",
        type=int,
        default=1,
        metavar="N",
        help="Stop after N runs pass enabled stall tests (default 1)",
    )
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
        "--tags-mask",
        nargs="*",
        default=None,
        metavar="MASK",
        help=(
            "Per-statistic mask suffixes (one per statistic in --tags_stat). "
            "Default: tags_mask_for_sweep (bispec → _kb0.25, else empty). "
            "Fiducial pk+bispec+pgm: --tags-mask '' _kb0.25 _kpgm0.25"
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Ignore best_run_log.csv resume state and re-run stall tests from scratch",
    )
    p.add_argument(
        "--shame-tag-mock",
        default="_nbar0.00022",
        metavar="TAG",
        help="SHAMe OOD mock tag for the second smoke test (default _nbar0.00022)",
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
    if args.n_passing < 1:
        raise ValueError("--n-passing must be >= 1, got %r" % (args.n_passing,))
    if args.n_samples_shame < 0:
        raise ValueError(
            "--n-samples-shame must be >= 0, got %r" % (args.n_samples_shame,)
        )
    skip_shame = args.n_samples_shame == 0

    sweep_root = sweep_root_for_tag_inf(tag_inf)
    log_path = sweep_root / BEST_RUN_LOG_CSV
    id_path = sweep_root / BEST_RUN_ID_TXT
    runs_path = sweep_root / BEST_RUNS_TXT
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
        clear_best_runs_ids(runs_path)
        print(
            "  overwrite: ignoring resume; cleared %s and %s"
            % (id_path, runs_path),
            flush=True,
        )
    else:
        saved = read_best_run_log(log_path)
        cur_ids = {rd.name for rd, _, _ in ranked}
        saved = {k: v for k, v in saved.items() if k in cur_ids}

    rows = merge_rows_for_write(ranked, saved, skip_shame=skip_shame)
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
    y_shame_ood: list[np.ndarray] = []
    if skip_shame:
        print(
            "  Smoke tests: coverage only (batch %d, n_samples=%d); SHAMe disabled"
            % (args.stall_batch_size, args.n_samples),
            flush=True,
        )
    else:
        y_shame_ood = load_shame_ood_y_obs(
            statistics, tags_mask, args.shame_tag_mock
        )
        print(
            "  Smoke tests: coverage (batch %d, n_samples=%d) + SHAMe OOD %s (n_samples=%d)"
            % (
                args.stall_batch_size,
                args.n_samples,
                args.shame_tag_mock,
                args.n_samples_shame,
            ),
            flush=True,
        )

    n_target = args.n_passing
    n_pass = count_passing_rows(rows)
    if n_pass >= n_target:
        chosen = write_best_run_outputs(id_path, runs_path, ranked, rows, n_target)
        print(
            "  Already have %d passing run(s) (target %d); wrote %s and %s: %s"
            % (n_pass, n_target, id_path.name, runs_path.name, ", ".join(chosen)),
            flush=True,
        )
        return

    print("  Target: %d passing runs (have %d so far)" % (n_target, n_pass), flush=True)

    for i, (rd, mv, _lm) in enumerate(ranked):
        rid = rd.name
        idx = next(j for j, x in enumerate(rows) if x[0] == rid)
        _, _, p_cov, p_shame, pcur = rows[idx]
        scored = (
            p_cov in ("true", "false")
            and (skip_shame or p_shame in ("true", "false"))
            and pcur in ("true", "false")
        )
        if scored:
            print(
                "  [%d] %s  loss=%s  coverage=%s shame=%s pass=%s (resume, skip)"
                % (i + 1, rid, mv, p_cov, p_shame, pcur),
                flush=True,
            )
            if pcur == "false":
                continue
        else:
            print("  [%d] smoke test %s ..." % (i + 1, rid), flush=True)
            res = run_dual_smoke_eval(
                sweep_root,
                rd,
                statistics,
                y_cov_nb,
                y_shame_ood,
                args.stall_batch_size,
                args.batch_timeout_seconds,
                args.n_samples,
                args.n_samples_shame,
            )
            p_cov = res["pass_coverage"]
            p_shame = res["pass_shame"]
            pnew = "true" if res.get("status") == "ok" else "false"
            rows[idx] = (rid, mv, p_cov, p_shame, pnew)
            write_best_run_log(log_path, rows)
            print(
                "    -> coverage=%s  shame=%s  pass=%s  wrote %s"
                % (res["coverage"], res["shame"], pnew, log_path),
                flush=True,
            )
            pcur = pnew

        n_pass = count_passing_rows(rows)
        if pcur == "true":
            chosen = write_best_run_outputs(id_path, runs_path, ranked, rows, n_target)
            print(
                "  passing %d/%d  best_run=%s  best_runs=%s"
                % (n_pass, n_target, chosen[0] if chosen else "?", ", ".join(chosen)),
                flush=True,
            )
        if n_pass >= n_target:
            print(
                "  Done: %d passing runs (target %d)." % (n_pass, n_target),
                flush=True,
            )
            return

    n_pass = count_passing_rows(rows)
    if n_pass:
        chosen = write_best_run_outputs(id_path, runs_path, ranked, rows, n_target)
        print(
            "  Exhausted ranked list with %d passing run(s) (target %d): %s"
            % (n_pass, n_target, ", ".join(chosen)),
            flush=True,
        )
    else:
        clear_best_run_id(id_path)
        clear_best_runs_ids(runs_path)
        print(
            "  No run passed stall test (cleared %s and %s)."
            % (id_path.name, runs_path.name),
            flush=True,
        )


def main() -> int:
    args = parse_args()
    try:
        statistics, tags_mask = parse_tags_stat(args.tags_stat)
    except ValueError as e:
        sys.stderr.write("tags_stat: %s\n" % e)
        return 1
    tags_stat_display = "_" + "_".join(statistics)
    if args.tags_mask is not None:
        tags_mask = list(args.tags_mask)
        if len(tags_mask) != len(statistics):
            sys.stderr.write(
                "--tags-mask has %d value(s) but --tags_stat has %d statistic(s)\n"
                % (len(tags_mask), len(statistics))
            )
            return 1

    for noise_mode in args.noise_modes:
        process_one_combo(
            args, noise_mode, statistics, tags_mask, tags_stat_display
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
