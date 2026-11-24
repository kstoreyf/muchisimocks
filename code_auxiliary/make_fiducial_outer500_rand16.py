"""make_fiducial_outer500_rand16.py

Full 1000-row coverage for the fiducial top-model networks of every statistics combination (user, 2026-11-23):
outer500 coverage (test only) for each top network that has inner500 but no full-set file; the full files are then
written by merge_coverage_subsets.py, which keeps the subset files too (both are on disk; nothing reads halves).
Top networks = setting 1, seed 0 (the copied sweep trial, top3_settings) and seed 1 (the fresh retrain, top3_fresh).

Lists, by combination, in jobs/lists/fiducial_outer500_<stats>_rand16.txt. P+P_gm+B's are in make_conv_grid_rand16.py
(fiducial_outer500_top_rand16.txt, submitted first). P+P_gm needs its kpgm0.37 sweep (running) first: rerun this
then, with --stats pk_pgm.

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_fiducial_outer500_rand16.py [--stats pk pk_bispec]
  sbatch --job-name=test_fiducial_outer500_<stats> --array=0-<n-1> jobs/job_train_v2.s jobs/lists/<list>.txt
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fresh_fid_rand16 as mff  # noqa: E402  (sets sys.path / cwd for code/)
import generate_config_inference as gci  # noqa: E402
import paths  # noqa: E402

STATS = {"pk": mff.P, "pk_bispec": mff.PB, "pk_pgm": ["pk", "pgm"]}
TAG_TEST_COV = "_coveragev2_p5_n1000_biasnoisecoveragev2_p9_n1000_noisefly"
TAIL = "_p5_n10000_biasnoisenestv2_p9_n320000_noisefly_rp_bx32_ntrain10000"


def needs_outer(stats, masks, seed):
    """True if the network exists with inner500 coverage and has neither the full nor the outer500 file."""
    d = (Path(paths.DIR_RESULTS) / "results_sbi" / f"sbi_muchisimocks_{'_'.join(stats)}{''.join(masks)}{TAIL}"
         f"{gci.best_tag(gci.TAG_SWEEP, None, stats)}{'_seed1' if seed else ''}")
    f = d / f"samples_test_muchisimocks_{'_'.join(stats)}{''.join(masks)}{TAG_TEST_COV}"
    has = lambda sub: Path(f"{f}{sub}_pred.npy").is_file()
    return d.is_dir() and has("_inner500") and not has("") and not has("_outer500")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", nargs="+", default=["pk", "pk_bispec"], choices=list(STATS))
    a = ap.parse_args()
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    for name in a.stats:
        stats = STATS[name]
        masks = gci.tags_mask_for_sweep(stats)
        seeds = [s for s in (0, 1) if needs_outer(stats, masks, s)]
        lines = [mff.line(stats, masks, 0, s, kw, train=False, ood=False, coverage=True, coverage_subset="outer500")
                 for s in seeds]
        fn = mff.DIR_LISTS / f"fiducial_outer500_{name}_rand16.txt"
        if lines:
            fn.write_text("\n".join(lines) + "\n")
        print(f"{name}: outer500 needed for seeds {seeds}" + (f" -> {fn}" if lines else ""))


if __name__ == "__main__":
    main()
