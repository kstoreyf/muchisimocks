"""make_ensemble_tests_rand16.py

Ensemble comparison for the v2 -rand16 sweeps (fiducial cuts), every statistic combo:
  - top-3 hyperparameter settings (best_runs.txt lines 0, 1, 2; seed 0 = the sweep trial, copied)
  - top setting x seeds 0, 1, 2 (seeds 1, 2 trained by make_best_seeds_list_v2.py)
Tests per model: coverage inner500 (idxs_obs_coveragev2_p5_n1000_inner500.npy; the top setting, seed 0
runs all 1000 rows instead) and SHAMe at the 3 densities.

The top setting's 3 seeds are trained by the bestseeds_v2_<combo> jobs, so their lines use train config
"none" (test only). The full combo's seeds were still training when this was written: its lines go in a
separate list, submitted with --dependency on that job.

Also writes ens_rand16_topextra.txt: the remaining fiducial tests for the top setting, seed 0 (in the
ensemble whichever way we choose): CV set (mean), phase-matched mocks at the 3 densities (individual + mean).

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_ensemble_tests_rand16.py
  sbatch --job-name=test_ens_rand16 --array=0-16 jobs/job_train_v2.s jobs/lists/ens_rand16.txt
  sbatch --job-name=test_ens_rand16_full --dependency=afterok:<bestseeds job> --array=0-2 \
      jobs/job_train_v2.s jobs/lists/ens_rand16_fullseeds.txt
  sbatch --job-name=test_top_rand16_cv_matched --array=0-3 jobs/job_train_v2.s jobs/lists/ens_rand16_topextra.txt
      (task 3, the full combo, after the bestseeds job: --dependency)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIR_CODE = HERE.parent / "code"
sys.path.insert(0, str(DIR_CODE))
os.chdir(DIR_CODE)   # configs are written with paths relative to code/

import generate_config_inference as gci  # noqa: E402

TAG_SWEEP = "-rand16"
COMBOS = [["pk"], ["pk", "pgm"], ["pk", "bispec"], ["pk", "pgm", "bispec"]]   # sweep dir names
FULL_COMBO = ["pk", "pgm", "bispec"]
MODELS = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]   # (nth_best_run, seed)
COVERAGE_SUBSET = "inner500"
DIR_LISTS = HERE / "jobs" / "lists"


def rel(fn):
    return os.path.relpath(fn, DIR_CODE)


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    train_kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines, lines_full, lines_extra = [], [], []
    for statistics in COMBOS:
        tags_mask = [gci.SWEEP_MASKS[s] for s in statistics]
        for nth, seed in MODELS:
            nth_arg = None if nth == 0 else nth
            if nth == 0:
                fn_train = "none"   # trained by the bestseeds_v2 jobs
            else:
                fn_train = rel(gci.generate_train_config(
                    overwrite=False, statistics=statistics, tags_mask=tags_mask, run_mode="best",
                    tag_sweep=TAG_SWEEP, nth_best_run=nth, seed=seed, **train_kw))
            common = dict(tag_params="_p5_n10000", noise_mode="noisy", overwrite=True,
                          statistics=statistics, tags_mask=tags_mask, tag_sweep=TAG_SWEEP,
                          nth_best_run=nth_arg, seed=seed)
            tests = [gci.generate_test_config_from_preset("ood", tag_mock=t, **common)
                     for t in gci.NBAR_TAGS_SHAME]
            # the top setting, seed 0 runs the full coverage set (Fig 4/5); its inner500 rows are taken from it
            tests.append(gci.generate_test_config_from_preset(
                "coverage", idxs_obs_subset=None if (nth, seed) == (0, 0) else COVERAGE_SUBSET, **common))
            for fn in tests:
                assert Path(fn).is_file(), fn
            line = " ".join([fn_train] + [rel(f) for f in tests])
            (lines_full if (statistics == FULL_COMBO and nth == 0) else lines).append(line)
            if (nth, seed) == (0, 0):
                extra = [gci.generate_test_config_from_preset("fixed_cosmo_shame_mean", **common)]
                for nbar_tag in gci.NBAR_TAGS_SHAME:
                    nbar = nbar_tag[len("_nbar"):]
                    for mean in (False, True):
                        extra.append(gci.generate_test_config(
                            overwrite=True, statistics=statistics, tags_mask=tags_mask, tag_sweep=TAG_SWEEP,
                            seed=seed, **train_kw, evaluate_mean=mean,
                            tag_params_test=f"_shamematched{nbar}_p0_n10",
                            tag_biasparams_test=f"_biasshame{nbar}_noisebest_p0_n1",
                            tag_noise_test="_noisematched"))
                lines_extra.append(" ".join(["none"] + [rel(f) for f in extra]))
    for name, ll in (("ens_rand16", lines), ("ens_rand16_fullseeds", lines_full),
                     ("ens_rand16_topextra", lines_extra)):
        fn = DIR_LISTS / f"{name}.txt"
        fn.write_text("\n".join(ll) + "\n")
        print(f"wrote {fn} ({len(ll)} lines)")


if __name__ == "__main__":
    main()
