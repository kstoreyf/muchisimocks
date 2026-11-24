"""make_conv_grid_rand16.py

Training-set convergence grid (paper Figs 10-11) for the TOP model of P+P_gm+B at the new fiducial
(kp0.37 / kpgm0.37 / kb0.24, ``_best-rand16-topkpgm0.37``, setting 1, seed 0), agreed 2026-11-23 for a first
version of the plots. Grid = plotter_figs.n_cosmo_arr x bx_arr (7 x 6). The (10000, 32) point is the fiducial
network itself, so it is not retrained.

Per grid network: training + coverage on the inner500 rows (plotter_figs.CONV_COVERAGE_SUBSET, what Figs 10-11 read)
+ SHAMe at the 3 densities.
Second list (fiducial_outer500_top_rand16.txt), test only: outer500 coverage of the two fiducial top-model networks,
seed 0 (the copied sweep trial, top3_settings scheme) and seed 1 (the fresh retrain, top3_fresh). They already have
inner500; merge_coverage_subsets.py then writes their full 1000-row coverage files (no rows are rerun).
Third list (fiducial_cv_top_rand16.txt), test only: the cosmic-variance set (_cvshamev2_p0_n1000) for the fiducial
top model, the mean (Figs 3, B1), seeds 0 and 1. (App A's 1000 individual realizations: deferred, user 2026-11-23.)

kb0.24 is still provisional (notebooks/2026-11-19_v2_checks.ipynb B3/B4).

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_conv_grid_rand16.py
  sbatch --job-name=train_conv_grid_top_rand16 --array=0-<n-1> jobs/job_train_v2.s jobs/lists/conv_grid_top_rand16.txt
  sbatch --job-name=test_fiducial_outer500_top --array=0-1 jobs/job_train_v2.s jobs/lists/fiducial_outer500_top_rand16.txt
  sbatch --job-name=test_fiducial_cv_top --array=0-1 jobs/job_train_v2.s jobs/lists/fiducial_cv_top_rand16.txt
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fresh_fid_rand16 as mff  # noqa: E402  (sets sys.path / cwd for code/)
import generate_config_inference as gci  # noqa: E402
import plotter_figs  # noqa: E402

N_COSMO, BX = plotter_figs.n_cosmo_arr, plotter_figs.bx_arr


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    assert gci.SWEEP_MASKS["pgm"] == "_kpgm0.37", "expects the kpgm0.37 fiducial"
    kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    masks = gci.tags_mask_for_sweep(mff.PPB)
    assert masks == ["_kp0.37", "_kpgm0.37", "_kb0.24"], masks
    fid = (gci.N_TRAIN_SWEEP, gci.BX_SWEEP)
    # largest training sets first: they take longest
    grid = sorted(((n, bx) for n in N_COSMO for bx in BX if (n, bx) != fid), key=lambda t: -t[0] * t[1])
    assert plotter_figs.CONV_COVERAGE_SUBSET == "inner500"
    lists = {
        "conv_grid_top_rand16": [mff.line(mff.PPB, masks, 0, 0, kw, matched=False, coverage=True,
                                          coverage_subset=plotter_figs.CONV_COVERAGE_SUBSET, n_train=n, bx=bx)
                                 for n, bx in grid],
        "fiducial_outer500_top_rand16": [mff.line(mff.PPB, masks, 0, seed, kw, train=False, ood=False, coverage=True,
                                                  coverage_subset="outer500") for seed in (0, 1)],
        # cosmic-variance set, mean only (Figs 3, B1; cheap). The 1000 individual realizations (App A) are deferred.
        "fiducial_cv_top_rand16": [
            "none " + mff.rel(gci.generate_test_config_from_preset(
                "fixed_cosmo_shame_mean", tag_params="_p5_n10000", noise_mode="noisy", overwrite=True,
                statistics=mff.PPB, tags_mask=masks, tag_sweep=mff.TAG_SWEEP, nth_best_run=None, seed=seed))
            for seed in (0, 1)],
    }
    for name, ll in lists.items():
        fn = mff.DIR_LISTS / f"{name}.txt"
        fn.write_text("\n".join(ll) + "\n")
        print(f"wrote {fn} ({len(ll)} lines)")


if __name__ == "__main__":
    main()
