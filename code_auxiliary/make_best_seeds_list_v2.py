"""make_best_seeds_list_v2.py

Write v2 train configs for extra seeds of a finished sweep's best model (best_runs.txt line 0, from
choose_best_run.py), and a job_train_v2.s list file with one line per config.

Seed 0 is the sweep trial itself (run_inference.py copies it, no training), so the nbest0 seed set
is complete for ensembling; seeds 1.. retrain the same hyperparameters with a new seed.

Run with code_auxiliary/env_v2.sh sourced (seconds; login node is fine):
  python make_best_seeds_list_v2.py pk pgm [--seeds 0 1 2] [--nth 0]
Then (from code_auxiliary/):
  sbatch --job-name=train_v2_bestseeds_pk_pgm --array=0-2 jobs/job_train_v2.s jobs/lists/bestseeds_v2_pk_pgm.txt
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
os.chdir(HERE.parent / "code")   # configs are written with paths relative to code/

from generate_config_inference import (generate_train_config, resolve_train_tag_bundle,
                                       SWEEP_MASKS, TAG_SWEEP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("statistics", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--nth", type=int, default=0, help="line of best_runs.txt (0 = best)")
    args = ap.parse_args()
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"

    train = resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = []
    for seed in args.seeds:
        fn = generate_train_config(overwrite=False, statistics=args.statistics,
                                   tags_mask=[SWEEP_MASKS[s] for s in args.statistics],
                                   run_mode="best", tag_sweep=TAG_SWEEP, nth_best_run=args.nth,
                                   seed=seed, **train)
        lines.append(os.path.relpath(fn, HERE.parent / "code"))
    tag = "_".join(args.statistics) + ("" if args.nth == 0 else f"_nbest{args.nth}")
    fn_list = HERE / "jobs" / "lists" / f"bestseeds_v2_{tag}.txt"
    fn_list.parent.mkdir(parents=True, exist_ok=True)
    fn_list.write_text("\n".join(lines) + "\n")
    print(f"wrote {fn_list} ({len(lines)} lines):")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
