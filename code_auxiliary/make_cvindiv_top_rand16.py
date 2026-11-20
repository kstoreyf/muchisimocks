"""make_cvindiv_top_rand16.py

Per-realization CV-set test (`_cvshamev2_p0_n1000`, all 1000 mocks at the SHAMe cosmology, no `_mean`) for the
top -rand16 setting, seed 0 (member 0 of either ensemble scheme), at the fiducial cuts. Paper Fig A1 uses it,
for P + P_gm + B only, so that's the only combo here. The model already exists, so the train entry is "none".

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_cvindiv_top_rand16.py
  sbatch --job-name=test_top_rand16_cvindiv --array=0 jobs/job_train_v2.s jobs/lists/cvindiv_top_rand16.txt
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
STATS_LIST = [["pk", "pgm", "bispec"]]
FN_LIST = HERE / "jobs" / "lists" / "cvindiv_top_rand16.txt"


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    lines = []
    for statistics in STATS_LIST:
        tags_mask = [gci.SWEEP_MASKS[s] for s in statistics]
        fn = gci.generate_test_config_from_preset(
            "fixed_cosmo_shame_sample", tag_params="_p5_n10000", noise_mode="noisy", overwrite=True,
            statistics=statistics, tags_mask=tags_mask, tag_sweep=TAG_SWEEP, seed=0)
        assert Path(fn).is_file(), fn
        lines.append(f"none {os.path.relpath(fn, DIR_CODE)}")
    FN_LIST.write_text("\n".join(lines) + "\n")
    print(f"wrote {FN_LIST} ({len(lines)} lines)")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
