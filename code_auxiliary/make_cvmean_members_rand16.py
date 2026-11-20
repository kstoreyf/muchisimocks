"""make_cvmean_members_rand16.py

CV-mean test (`_cvshamev2_p0_n1000`, mean data vector at the SHAMe cosmology) for the 4 ensemble candidates
other than the top setting, seed 0 (which already has it): top setting seeds 1, 2 and settings nbest1, nbest2,
seed 0, for every statistic combo at the fiducial cuts. All models exist, so train entries are "none".
One line per combo, 4 single-obs tests each.

Also matchedmean_members_rand16.txt: phase-matched mock mean (`_shamematched<nbar>_p0_n10` ... `_noisematched`,
evaluate_mean) at the 3 densities for the same 4 candidates, P + P_gm + B only (paper Fig 7). One line per member.

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_cvmean_members_rand16.py
  sbatch --job-name=test_ens_rand16_cvmean --array=0-3 jobs/job_train_v2.s jobs/lists/cvmean_members_rand16.txt
  sbatch --job-name=test_ens_rand16_matchedmean --array=0-3 jobs/job_train_v2.s jobs/lists/matchedmean_members_rand16.txt
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
COMBOS = [["pk"], ["pk", "pgm"], ["pk", "bispec"], ["pk", "pgm", "bispec"]]
MODELS = [(0, 1), (0, 2), (1, 0), (2, 0)]   # (nth_best_run, seed)
FN_LIST = HERE / "jobs" / "lists" / "cvmean_members_rand16.txt"
FN_LIST_MATCHED = HERE / "jobs" / "lists" / "matchedmean_members_rand16.txt"
FULL_COMBO = ["pk", "pgm", "bispec"]


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    lines = []
    for statistics in COMBOS:
        tags_mask = [gci.SWEEP_MASKS[s] for s in statistics]
        tests = []
        for nth, seed in MODELS:
            fn = gci.generate_test_config_from_preset(
                "fixed_cosmo_shame_mean", tag_params="_p5_n10000", noise_mode="noisy", overwrite=True,
                statistics=statistics, tags_mask=tags_mask, tag_sweep=TAG_SWEEP,
                nth_best_run=None if nth == 0 else nth, seed=seed)
            assert Path(fn).is_file(), fn
            tests.append(os.path.relpath(fn, DIR_CODE))
        lines.append(" ".join(["none"] + tests))
    FN_LIST.write_text("\n".join(lines) + "\n")
    print(f"wrote {FN_LIST} ({len(lines)} lines)")

    train_kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    tags_mask = [gci.SWEEP_MASKS[s] for s in FULL_COMBO]
    lines = []
    for nth, seed in MODELS:
        tests = []
        for nbar_tag in gci.NBAR_TAGS_SHAME:
            nbar = nbar_tag[len("_nbar"):]
            fn = gci.generate_test_config(
                overwrite=True, statistics=FULL_COMBO, tags_mask=tags_mask, tag_sweep=TAG_SWEEP,
                nth_best_run=None if nth == 0 else nth, seed=seed, **train_kw, evaluate_mean=True,
                tag_params_test=f"_shamematched{nbar}_p0_n10",
                tag_biasparams_test=f"_biasshame{nbar}_noisebest_p0_n1",
                tag_noise_test="_noisematched")
            assert Path(fn).is_file(), fn
            tests.append(os.path.relpath(fn, DIR_CODE))
        lines.append(" ".join(["none"] + tests))
    FN_LIST_MATCHED.write_text("\n".join(lines) + "\n")
    print(f"wrote {FN_LIST_MATCHED} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
