"""make_overall_kmax_rand16.py

Models for paper Fig 9 (overall k_max), v2 with -rand16 hyperparameters. For each statistic combo and each
k in data_loader.K_CUTS, the joined mask is ensemble_kmax_utils.overall_k_mask(stats, k): every statistic cut
at k up to the fiducial B / P_gm cut (KCUT_FID_OTHER = 0.24); above it only P_gg moves, B and P_gm stay
at 0.24. The k = 0.37 point is the fiducial model (already trained), so it is skipped.

Each point is the top hyperparameter setting of the combo's fiducial-cut sweep, seed 0 (member 0 of either
ensemble scheme), retrained at the point's cut, like make_kmax_grid_rand16.py.
Tests per point: SHAMe at the 3 densities, phase-matched mock at the 3 densities (individual + mean).

Note: the k > 0.24 points assume the B / P_gm fiducial stays at 0.24. If the k_max grid moves it, those
points need retraining; the k <= 0.24 points don't depend on it.

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_overall_kmax_rand16.py
  sbatch --job-name=train_overallkmax_rand16 --array=0-<n-1> jobs/job_train_v2.s jobs/lists/overallkmax_rand16.txt
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIR_CODE = HERE.parent / "code"
sys.path.insert(0, str(DIR_CODE))
os.chdir(DIR_CODE)   # configs are written with paths relative to code/

import data_loader  # noqa: E402
import ensemble_kmax_utils as eku  # noqa: E402
import generate_config_inference as gci  # noqa: E402

TAG_SWEEP = "-rand16"
FN_LIST = HERE / "jobs" / "lists" / "overallkmax_rand16.txt"


def rel(fn):
    return os.path.relpath(fn, DIR_CODE)


def points():
    """(statistics, tags_mask) for every non-fiducial overall-k_max point."""
    out = []
    for statistics in eku.STAT_ROWS:
        fid = "".join(gci.FIDUCIAL_MASKS[s] for s in statistics)
        for k in data_loader.K_CUTS:
            mask = eku.overall_k_mask(statistics, k)
            if mask == fid:
                continue
            out.append((list(statistics), eku.parse_mask_to_tags_mask(statistics, mask)))
    return out


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    train_kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = []
    for statistics, tags_mask in points():
        fn_train = gci.generate_train_config(
            overwrite=False, statistics=statistics, tags_mask=tags_mask, run_mode="best",
            tag_sweep=TAG_SWEEP, seed=0, **train_kw)
        tests = [gci.generate_test_config_from_preset(
            "ood", tag_mock=t, tag_params="_p5_n10000", noise_mode="noisy", overwrite=True,
            statistics=statistics, tags_mask=tags_mask, tag_sweep=TAG_SWEEP, seed=0)
            for t in gci.NBAR_TAGS_SHAME]
        for nbar_tag in gci.NBAR_TAGS_SHAME:
            nbar = nbar_tag[len("_nbar"):]
            for mean in (False, True):
                tests.append(gci.generate_test_config(
                    overwrite=True, statistics=statistics, tags_mask=tags_mask, tag_sweep=TAG_SWEEP,
                    seed=0, **train_kw, evaluate_mean=mean,
                    tag_params_test=f"_shamematched{nbar}_p0_n10",
                    tag_biasparams_test=f"_biasshame{nbar}_noisebest_p0_n1",
                    tag_noise_test="_noisematched"))
        for fn in [fn_train] + tests:
            assert Path(fn).is_file(), fn
        lines.append(" ".join([rel(fn_train)] + [rel(f) for f in tests]))
    FN_LIST.write_text("\n".join(lines) + "\n")
    print(f"wrote {FN_LIST} ({len(lines)} points)")
    for statistics, tags_mask in points():
        print("  ", "_".join(statistics), "".join(tags_mask))


if __name__ == "__main__":
    main()
