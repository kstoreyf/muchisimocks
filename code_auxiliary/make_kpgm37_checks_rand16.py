"""make_kpgm37_checks_rand16.py

Checks for moving the fiducial k_max^{P_gm} from 0.24 to 0.37 (full combo P + P_gm + B, kp0.37 throughout),
using the -rand16 hyperparameters from the fiducial-cut (kpgm0.24, kb0.24) sweep.

1. kpgm slice, top-3 ensemble (kpgmslice_ens_rand16.txt): kb0.24, kpgm over data_loader.K_CUTS,
   settings 2 and 3 of the fiducial sweep (best_runs.txt lines 1, 2), seed 0. Setting 1 at every point is the
   k_max grid model (make_kmax_grid_rand16.py). At kpgm0.24 settings 2 and 3 are the sweep trials (copied).
2. Edge controls (edgectrl_rand16.txt): settings 2 and 3 at (kpgm0.37, kb0.11) and (kpgm0.37, kb0.37), the
   cuts of the edge sweeps (code/submit_scripts/submit_inf_train_sweep.sh with KCUTS=...). They give the
   hyperparameter + seed scatter for deciding whether the fiducial settings transfer to the kb edges.
   Setting 1 at these cuts is the grid model: its line is test only ("none").
3. kpgm0.24 retrains (kpgm24_retrain_rand16.txt): settings 2 and 3 at (kpgm0.24, kb0.24) with seed 1. At kpgm0.24 the
   slice members are the sweep trials themselves (selected on val loss, so flattered by the winner's curse); these
   give a retrained top-3 ensemble there (setting 1, seed 1 = the existing `_seed1` model) for a fair FoM comparison.
4. Failed-member retrains (slice_retrain_rand16.txt): rule agreed 2026-11-22, a slice member whose best val NLL is
   > RETRAIN_THR (0.4 nats) above the best member at its cut is a failed training and is retrained with seed 1
   (replacing seed 0 in the mixture). Flagged: setting 3 at kpgm 0.18, 0.21, 0.27 (notebook 2026-11-19 §6b).

Tests per model: SHAMe at the 3 densities, phase-matched mocks at the 3 densities (individual + mean).
Models at kpgm0.37 (slice point + edges) also run coverage inner500 (in-distribution calibration).

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_kpgm37_checks_rand16.py
  sbatch --job-name=train_kpgmslice_ens_rand16 --array=0-<n-1> jobs/job_train_v2.s jobs/lists/kpgmslice_ens_rand16.txt
  sbatch --job-name=train_edgectrl_rand16 --array=0-<n-1> jobs/job_train_v2.s jobs/lists/edgectrl_rand16.txt
  sbatch --job-name=train_kpgm24_retrain_rand16 --array=0-<n-1> jobs/job_train_v2.s jobs/lists/kpgm24_retrain_rand16.txt
  sbatch --job-name=train_slice_retrain_rand16 --array=0-<n-1> jobs/job_train_v2.s jobs/lists/slice_retrain_rand16.txt
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
STATS = ["pk", "pgm", "bispec"]
KP, KB_FID, KPGM_NEW = 0.37, 0.24, 0.37
KB_EDGES = (0.11, 0.37)
FLAGGED = [(0.18, 2), (0.21, 2), (0.27, 2)]   # (kpgm, nth setting) failing the 0.4-nat rule at kb0.24
COVERAGE_SUBSET = "inner500"
DIR_LISTS = HERE / "jobs" / "lists"


def rel(fn):
    return os.path.relpath(fn, DIR_CODE)


def masks(kpgm, kb):
    return [data_loader.kcut_tag("pk", KP), data_loader.kcut_tag("pgm", kpgm), data_loader.kcut_tag("bispec", kb)]


def model_line(kpgm, kb, nth, train_kw, coverage, seed=0):
    tags_mask = masks(kpgm, kb)
    nth_arg = None if nth == 0 else nth
    fn_train = "none" if nth == 0 else rel(gci.generate_train_config(
        overwrite=False, statistics=STATS, tags_mask=tags_mask, run_mode="best",
        tag_sweep=TAG_SWEEP, nth_best_run=nth_arg, seed=seed, **train_kw))
    common = dict(statistics=STATS, tags_mask=tags_mask, tag_sweep=TAG_SWEEP, nth_best_run=nth_arg, seed=seed)
    tests = []
    if nth != 0:   # setting 1 already has the SHAMe / matched tests from the grid
        tests += [gci.generate_test_config_from_preset("ood", tag_mock=t, tag_params="_p5_n10000",
                                                       noise_mode="noisy", overwrite=True, **common)
                  for t in gci.NBAR_TAGS_SHAME]
        for nbar_tag in gci.NBAR_TAGS_SHAME:
            nbar = nbar_tag[len("_nbar"):]
            for mean in (False, True):
                tests.append(gci.generate_test_config(
                    overwrite=True, **common, **train_kw, evaluate_mean=mean,
                    tag_params_test=f"_shamematched{nbar}_p0_n10",
                    tag_biasparams_test=f"_biasshame{nbar}_noisebest_p0_n1",
                    tag_noise_test="_noisematched"))
    if coverage:
        tests.append(gci.generate_test_config_from_preset(
            "coverage", tag_params="_p5_n10000", noise_mode="noisy", overwrite=True,
            idxs_obs_subset=COVERAGE_SUBSET, **common))
    for fn in tests:
        assert Path(fn).is_file(), fn
    return " ".join([fn_train] + [rel(f) for f in tests]) if tests else None


def main():
    # Written for the kpgm0.24-sweep models (plain "_best-rand16"); since 2026-11-22 the config generator names
    # P_gm combos after the kpgm0.37 sweep, so rerunning this would point at different models.
    assert gci.SWEEP_MASKS["pgm"] == "_kpgm0.24", "historical script: lists are for the kpgm0.24-sweep models"
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    train_kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    slice_lines = [model_line(kpgm, KB_FID, nth, train_kw, coverage=(kpgm == KPGM_NEW))
                   for kpgm in data_loader.K_CUTS for nth in (0, 1, 2)]
    edge_lines = [model_line(KPGM_NEW, kb, nth, train_kw, coverage=True)
                  for kb in KB_EDGES for nth in (0, 1, 2)]
    flagged_lines = [model_line(k, KB_FID, nth, train_kw, coverage=False, seed=1) for k, nth in FLAGGED]
    retrain_lines = [model_line(0.24, KB_FID, nth, train_kw, coverage=False, seed=1) for nth in (1, 2)]
    for name, ll in (("kpgmslice_ens_rand16", slice_lines), ("edgectrl_rand16", edge_lines),
                     ("kpgm24_retrain_rand16", retrain_lines), ("slice_retrain_rand16", flagged_lines)):
        ll = [l for l in ll if l]
        fn = DIR_LISTS / f"{name}.txt"
        fn.write_text("\n".join(ll) + "\n")
        print(f"wrote {fn} ({len(ll)} lines, {sum(l.startswith('none') for l in ll)} test-only)")


if __name__ == "__main__":
    main()
