"""retire_sweep_trial.py

Retire one trial of a v2 W&B sweep so the sweep can draw a replacement (agreed 2026-11-23 for the P+P_gm+B
kpgm0.37/kb0.24 sweep: trial 0jwzywhf, best_runs.txt line 2, fails the 0.4-nat rule at kb 0.27-0.37 in the kb slice
even after a seed-1 retrain, i.e. its hyperparameters don't transfer; notebooks/2026-11-19_v2_checks.ipynb §7, §9).

Nothing is deleted locally; everything is moved under <results_sbi>/retired_sweep_trials/<sweep dir>/:
  - the trial dir                          -> <run_id>/
  - best-mode dirs trained from it         -> best_dirs/<dir>   (stats prefix + best tag + _nbest{n}[_seed*], n = its
                                              best_runs.txt line, any cuts); they would otherwise be reused under
                                              the same name for the run that moves into line n
  - a row appended to retired_trials.csv  (run id, hyperparameters, best val loss, reason, moved dirs)
Then the run is deleted from the W&B sweep (run_cap counts runs present), the sweep is resumed if W&B had marked it
FINISHED (it does once run_cap is reached; an agent then gets no job), and best_runs.txt / best_run.txt are cleared so nothing picks up a stale ranking before choose_best_run.py is rerun.

Afterwards: rerun choose_best (jobs/job_choose_best_v2.s <tags_stat>) and add one trial with
  N_SWEEP_AGENTS=1 bash code/submit_scripts/submit_inf_train_sweep.sh <stats...>

Run with env_v2.sh sourced, benv (W&B API; seconds):
  python retire_sweep_trial.py --sweep-dir <sweep dir name> --run-id <id> --reason "<why>" [--dry-run]
"""
import argparse
import csv
import datetime
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import paths  # noqa: E402

DIR_SBI = Path(paths.DIR_RESULTS) / "results_sbi"
DIR_CONFIGS_TRAIN = HERE.parent / "configs" / "configs_train"
HP_KEYS = ("learning_rate", "hidden_features", "training_batch_size", "num_transforms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, help="sbi_muchisimocks_..._sweep-rand16 (name under results_sbi)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--best-tag", required=True, help="best-mode tag of this sweep, e.g. _best-rand16-topkpgm0.37")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"

    root = DIR_SBI / a.sweep_dir
    trial = root / a.run_id
    assert trial.is_dir(), trial
    best_runs = (root / "best_runs.txt").read_text().split()
    assert a.run_id in best_runs, f"{a.run_id} not in best_runs.txt {best_runs}"
    nth = best_runs.index(a.run_id)
    m = re.match(r"sbi_muchisimocks_(.+?)_kp", a.sweep_dir)
    stats = m.group(1)
    tail = a.sweep_dir[a.sweep_dir.index("_p5_"):a.sweep_dir.index("_sweep-")]
    suffix = a.best_tag + (f"_nbest{nth}" if nth else "")
    pat = re.compile(rf"^sbi_muchisimocks_{re.escape(stats)}_kp[^_]+(_k\w+?\d\.\d\d)*{re.escape(tail)}"
                     rf"{re.escape(suffix)}(_seed\d+)?$")
    best_dirs = sorted(p for p in DIR_SBI.iterdir() if p.is_dir() and pat.match(p.name))
    yaml = DIR_CONFIGS_TRAIN / f"config_muchisimocks{a.sweep_dir[len('sbi_muchisimocks'):]}.yaml"
    sweep_path = re.search(r"^wandb_sweep_id:\s*(\S+)", yaml.read_text(), re.M).group(1)

    print(f"sweep {sweep_path}\ntrial {a.run_id} = best_runs.txt line {nth} -> best-mode suffix {suffix}")
    print(f"{len(best_dirs)} best-mode dirs to move:\n  " + "\n  ".join(p.name for p in best_dirs))
    if a.dry_run:
        return

    import wandb
    run = wandb.Api().run(f"{sweep_path.rsplit('/', 1)[0]}/{a.run_id}")
    hp = {k: run.config.get(k) for k in HP_KEYS}
    best_val = run.summary.get("best_validation_loss")

    dest = DIR_SBI / "retired_sweep_trials" / a.sweep_dir
    (dest / "best_dirs").mkdir(parents=True, exist_ok=True)
    shutil.move(str(trial), str(dest / a.run_id))
    for p in best_dirs:
        shutil.move(str(p), str(dest / "best_dirs" / p.name))
    fn_log = root / "retired_trials.csv"
    new = not fn_log.is_file()
    with open(fn_log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "best_runs_line", *HP_KEYS, "best_validation_loss", "retired_at", "reason",
                        "moved_to", "n_best_dirs_moved"])
        w.writerow([a.run_id, nth, *hp.values(), best_val, datetime.datetime.now().isoformat(timespec="seconds"),
                    a.reason, str(dest), len(best_dirs)])
    for fn in ("best_runs.txt", "best_run.txt"):
        if (root / fn).is_file():
            (root / fn).rename(root / f"{fn}.before_retire_{a.run_id}")
    run.delete()
    import subprocess
    if wandb.Api().sweep(sweep_path).state.upper() == "FINISHED":
        subprocess.run(["wandb", "sweep", "--resume", sweep_path], check=True)
    print(f"moved trial + {len(best_dirs)} best dirs to {dest}; logged in {fn_log}; deleted W&B run {a.run_id} {hp}")


if __name__ == "__main__":
    main()
