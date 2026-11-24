"""make_pkpgm_fiducial_rand16.py

Fiducial P+P_gm networks at the new fiducial (kp0.37 / kpgm0.37, ``_best-rand16-topkpgm0.37``), for initial versions
of Figs 3-6, 10 (user, 2026-11-23). Submitted with a dependency on choose_best of the P+P_gm kpgm0.37 sweep (training
reads best_runs.txt at run time; seed 0 at the sweep cut is a copy of the rank-1 trial).

Top setting only: seed 0 (the copied sweep trial, top3_settings) and seed 1 (fresh retrain, top3_fresh). Tests:
SHAMe at the 3 densities, phase-matched mocks (individual + mean), the cosmic-variance-set mean, and coverage on
inner500 + outer500 (merged into the full 1000-row file afterwards by merge_coverage_subsets.py).

Second list (pkpgm_fiducial_nbest12_rand16.txt; user, 2026-11-24): settings 2-3 (nbest1, nbest2) at the fiducial cut.
Seed 0 = copies of the sweep trials (top3_settings; no training): SHAMe, matched, CV mean (no coverage, as for the other
combos' settings 2-3). Seed 1 = fresh retrains (top3_fresh, Figs 8-11): same tests + inner500 coverage (the conv grid's
fiducial point), as for the other combos.

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_pkpgm_fiducial_rand16.py
  sbatch --job-name=train_pkpgm_fiducial_rand16 --array=0-1 --dependency=afterok:<choosebest> \\
      jobs/job_train_v2.s jobs/lists/pkpgm_fiducial_rand16.txt
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fresh_fid_rand16 as mff  # noqa: E402  (sets sys.path / cwd for code/)
import generate_config_inference as gci  # noqa: E402

PPGM = ["pk", "pgm"]


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    masks = gci.tags_mask_for_sweep(PPGM)
    assert masks == ["_kp0.37", "_kpgm0.37"], masks
    kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = [mff.line(PPGM, masks, 0, seed, kw, cvmean=True, coverage=True, coverage_subset=("inner500", "outer500"))
             for seed in (0, 1)]
    fn = mff.DIR_LISTS / "pkpgm_fiducial_rand16.txt"
    fn.write_text("\n".join(lines) + "\n")
    print(f"wrote {fn} ({len(lines)} lines)")
    lines2 = [mff.line(PPGM, masks, nth, seed, kw, cvmean=True, coverage=bool(seed), coverage_subset="inner500")
              for seed in (0, 1) for nth in (1, 2)]
    fn2 = mff.DIR_LISTS / "pkpgm_fiducial_nbest12_rand16.txt"
    fn2.write_text("\n".join(lines2) + "\n")
    print(f"wrote {fn2} ({len(lines2)} lines)")


if __name__ == "__main__":
    main()
