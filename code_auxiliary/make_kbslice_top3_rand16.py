"""make_kbslice_top3_rand16.py

1D k_max^B slice with top-3 hyperparameter ensembles (kp0.37 throughout), to choose the fiducial k_max^B.
Agreed 2026-11-22 after the kpgm slice (notebooks/2026-11-19_v2_checks.ipynb §6) settled k_max^{P_gm} = 0.37.

- P+P_gm+B at kpgm0.37: top 3 settings of the kpgm0.37/kb0.24 sweep (best_runs.txt lines 0-2, model names
  ``_best-rand16-topkpgm0.37[_nbest{n}]``), all 9 kb in data_loader.K_CUTS.
- P+B: top 3 settings of the P+B sweep (kb0.24, unchanged; ``_best-rand16[_nbest{n}]``). Setting 1 seed 0 at every
  kb != 0.24 is the k_max grid model (make_kmax_grid_rand16.py; tests done), so only settings 2 and 3 are trained there.

Every slice point is a fresh training, never the sweep trial itself: at the sweep cut (kb0.24) the slice members are
seed 1 of each setting (the sweep trials are the val-loss winners and are flattered by it). For P+B setting 1 seed 1
at kb0.24 already exists (the §2 3-seed ensemble). For P+P_gm+B the seed-0 models at kb0.24 (= the copied sweep
trials, the paper's fiducial ensemble) are also set up here, with coverage inner500.

Tests per model: SHAMe at the 3 densities, phase-matched mocks at the 3 densities (individual + mean).

Run with env_v2.sh sourced (seconds; login node is fine), after choose_best_run.py has written best_runs.txt for the
kpgm0.37 sweep:
  python make_kbslice_top3_rand16.py
  sbatch --job-name=train_kbslice_top3_<combo> --array=0-<n-1> jobs/job_train_v2.s jobs/lists/kbslice_top3_<combo>.txt
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

TAG_SWEEP = gci.TAG_SWEEP
KP, KPGM = 0.37, 0.37
KB_SWEEP = 0.24
COVERAGE_SUBSET = "inner500"
DIR_LISTS = HERE / "jobs" / "lists"
COMBOS = {"pk_pgm_bispec": ["pk", "pgm", "bispec"], "pk_bispec": ["pk", "bispec"]}


def rel(fn):
    return os.path.relpath(fn, DIR_CODE)


def masks(stats, kb):
    cut = {"pk": KP, "pgm": KPGM, "bispec": kb}
    return [data_loader.kcut_tag(s, cut[s]) for s in stats]


def model_line(stats, kb, nth, seed, train_kw, coverage=False):
    """Train config (best mode, nth setting, seed) + its SHAMe / matched tests (+ coverage)."""
    tags_mask = masks(stats, kb)
    nth_arg = None if nth == 0 else nth
    fn_train = gci.generate_train_config(overwrite=False, statistics=stats, tags_mask=tags_mask, run_mode="best",
                                         tag_sweep=TAG_SWEEP, nth_best_run=nth_arg, seed=seed, **train_kw)
    common = dict(statistics=stats, tags_mask=tags_mask, tag_sweep=TAG_SWEEP, nth_best_run=nth_arg, seed=seed)
    tests = [gci.generate_test_config_from_preset("ood", tag_mock=t, tag_params="_p5_n10000", noise_mode="noisy",
                                                  overwrite=True, **common) for t in gci.NBAR_TAGS_SHAME]
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
    for fn in [fn_train] + tests:
        assert Path(fn).is_file(), fn
    return " ".join(rel(f) for f in [fn_train] + tests)


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    assert gci.SWEEP_MASKS["pgm"] == data_loader.kcut_tag("pgm", KPGM), "expects the kpgm0.37 sweep as fiducial"
    train_kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = {
        "pk_pgm_bispec": [model_line(COMBOS["pk_pgm_bispec"], kb, nth, int(kb == KB_SWEEP), train_kw)
                          for kb in data_loader.K_CUTS for nth in (0, 1, 2)]
        # fiducial ensemble: the copied sweep trials (seed 0), with coverage
        + [model_line(COMBOS["pk_pgm_bispec"], KB_SWEEP, nth, 0, train_kw, coverage=True) for nth in (0, 1, 2)],
        "pk_bispec": [model_line(COMBOS["pk_bispec"], kb, nth, int(kb == KB_SWEEP), train_kw)
                      for kb in data_loader.K_CUTS for nth in (1, 2)],
    }
    for combo, ll in lines.items():
        fn = DIR_LISTS / f"kbslice_top3_{combo}.txt"
        fn.write_text("\n".join(ll) + "\n")
        print(f"wrote {fn} ({len(ll)} lines)")


if __name__ == "__main__":
    main()
