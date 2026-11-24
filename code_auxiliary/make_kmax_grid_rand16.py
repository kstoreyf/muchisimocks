"""make_kmax_grid_rand16.py

k_max grid for choosing the fiducial cuts (v2, -rand16 hyperparameters), with k_max^P fixed at 0.37:
  - P + P_gm + B: full 2D grid of kb x kpgm over data_loader.K_CUTS (0.11 ... 0.37), 81 points
  - P + B: kb over K_CUTS;  P + P_gm: kpgm over K_CUTS (9 points each)
Each point is the top hyperparameter setting of that combo's fiducial-cut sweep, seed 0, retrained at the
point's cut (best mode retrains whenever tags_mask != SWEEP_MASKS). The fiducial point itself is the
existing best-rand16 model (trained / copied by the bestseeds_v2 jobs): its line has train config "none".
Statistic order follows the sweep dirs: pk_pgm_bispec (masks _kp_kpgm_kb).

Tests per point: SHAMe at the 3 densities, phase-matched mock at the 3 densities (individual + mean).

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_kmax_grid_rand16.py
  sbatch --job-name=train_kmaxgrid_rand16 --array=0-<n-1> jobs/job_train_v2.s jobs/lists/kmaxgrid_rand16.txt
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIR_CODE = HERE.parent / "code"
sys.path.insert(0, str(DIR_CODE))
os.chdir(DIR_CODE)   # configs are written with paths relative to code/

import data_loader  # noqa: E402
import generate_config_inference as gci  # noqa: E402

TAG_SWEEP = "-rand16"
KP = 0.37
FN_LIST = HERE / "jobs" / "lists" / "kmaxgrid_rand16.txt"


def rel(fn):
    return os.path.relpath(fn, DIR_CODE)


def grid():
    """(statistics, tags_mask) for every grid point."""
    kp = data_loader.kcut_tag("pk", KP)
    out = []
    for k in data_loader.K_CUTS:
        out.append((["pk", "bispec"], [kp, data_loader.kcut_tag("bispec", k)]))
        out.append((["pk", "pgm"], [kp, data_loader.kcut_tag("pgm", k)]))
    for kpgm in data_loader.K_CUTS:
        for kb in data_loader.K_CUTS:
            out.append((["pk", "pgm", "bispec"],
                        [kp, data_loader.kcut_tag("pgm", kpgm), data_loader.kcut_tag("bispec", kb)]))
    return out


def main():
    # Written for the kpgm0.24-sweep models (plain "_best-rand16"); since 2026-11-22 the config generator names
    # P_gm combos after the kpgm0.37 sweep, so rerunning this would point at different models.
    assert gci.SWEEP_MASKS["pgm"] == "_kpgm0.24", "historical script: lists are for the kpgm0.24-sweep models"
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    train_kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = []
    for statistics, tags_mask in grid():
        fiducial = tags_mask == [gci.SWEEP_MASKS[s] for s in statistics]
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
        lines.append(" ".join(["none" if fiducial else rel(fn_train)] + [rel(f) for f in tests]))
    FN_LIST.write_text("\n".join(lines) + "\n")
    print(f"wrote {FN_LIST} ({len(lines)} points, {sum(l.startswith('none') for l in lines)} fiducial)")


if __name__ == "__main__":
    main()
