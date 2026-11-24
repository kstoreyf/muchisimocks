"""make_fresh_fid_rand16.py

Three small batches agreed 2026-11-22 (notebooks/2026-11-19_v2_checks.ipynb §7, 2026-11-22_paper_figures.ipynb):

1. kb-slice failed-member retrains (kbslice_retrain_rand16.txt). Standing rule: a member whose best val NLL is more than
   0.4 nats above the best member at its cut is a failed training; it is retrained with the next seed, which replaces it.
   Flagged in §7: P+B setting 2 at kb0.24 (seed 1 -> seed 2), P+B setting 1 at kb0.31 (grid model, seed 0 -> 1),
   P+P_gm+B setting 3 at kb0.37 (seed 0 -> 1).
2. Fresh fiducial members for P alone (fresh_pk_rand16.txt): settings 2 and 3, seed 1, at kp0.37. The `top3_fresh`
   ensemble (ensemble_kmax_utils) uses seed-1 retrains wherever a member would be a sweep trial; setting 1 seed 1
   already exists (the §2 3-seed ensemble).
3. Coverage (inner500) of the fresh fiducial members (fresh_coverage_rand16.txt, test only): P+B and P+P_gm+B settings
   2, 3 seed 1 (setting 1 seed 1 of P / P+B has it from §2; P+P_gm+B setting 1 seed 1 does not). Item 2's P members
   get coverage in their own lines. The outer 500 rows are to follow (needs a complement idxs subset).

Tests on trained models: SHAMe at the 3 densities, phase-matched mocks at the 3 densities (individual + mean).

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_fresh_fid_rand16.py
  sbatch --job-name=train_<list name> --array=0-<n-1> jobs/job_train_v2.s jobs/lists/<list name>.txt
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIR_CODE = HERE.parent / "code"
sys.path.insert(0, str(DIR_CODE))
os.chdir(DIR_CODE)   # configs are written with paths relative to code/

import generate_config_inference as gci  # noqa: E402

TAG_SWEEP = gci.TAG_SWEEP
COVERAGE_SUBSET = "inner500"
DIR_LISTS = HERE / "jobs" / "lists"
P, PB, PPB = ["pk"], ["pk", "bispec"], ["pk", "pgm", "bispec"]


def rel(fn):
    return os.path.relpath(fn, DIR_CODE)


def line(stats, tags_mask, nth, seed, train_kw, *, train=True, ood=True, matched=True, cvmean=False,
         coverage=False, coverage_subset=COVERAGE_SUBSET, n_train=None, bx=None):
    """Train config (or "none" = test only) + its tests. coverage_subset: a subset name, None (all 1000 coverage
    rows) or a tuple of subset names (one coverage test each, e.g. ("inner500", "outer500")).
    cvmean: the cosmic-variance-set mean test (fixed_cosmo_shame_mean).
    n_train / bx: training-set size (default: the sweep's, N_TRAIN_SWEEP / BX_SWEEP)."""
    nth_arg = None if nth == 0 else nth
    common = dict(statistics=stats, tags_mask=tags_mask, tag_sweep=TAG_SWEEP, nth_best_run=nth_arg, seed=seed)
    if n_train is not None:
        common.update(n_train=n_train, bx=bx)
    fn_train = (gci.generate_train_config(overwrite=False, run_mode="best", **common, **train_kw) if train else None)
    tests = []
    if ood:
        tests += [gci.generate_test_config_from_preset("ood", tag_mock=t, tag_params="_p5_n10000", noise_mode="noisy",
                                                       overwrite=True, **common) for t in gci.NBAR_TAGS_SHAME]
        for nbar_tag in (gci.NBAR_TAGS_SHAME if matched else []):
            nbar = nbar_tag[len("_nbar"):]
            for mean in (False, True):
                tests.append(gci.generate_test_config(
                    overwrite=True, **common, **train_kw, evaluate_mean=mean,
                    tag_params_test=f"_shamematched{nbar}_p0_n10",
                    tag_biasparams_test=f"_biasshame{nbar}_noisebest_p0_n1", tag_noise_test="_noisematched"))
    if cvmean:
        tests.append(gci.generate_test_config_from_preset(
            "fixed_cosmo_shame_mean", tag_params="_p5_n10000", noise_mode="noisy", overwrite=True, **common))
    if coverage:
        for sub in (coverage_subset if isinstance(coverage_subset, tuple) else (coverage_subset,)):
            tests.append(gci.generate_test_config_from_preset(
                "coverage", tag_params="_p5_n10000", noise_mode="noisy", overwrite=True,
                idxs_obs_subset=sub, **common))
    for fn in ([fn_train] if fn_train else []) + tests:
        assert Path(fn).is_file(), fn
    return " ".join([rel(fn_train) if fn_train else "none"] + [rel(f) for f in tests])


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    assert gci.SWEEP_MASKS["pgm"] == "_kpgm0.37", "expects the kpgm0.37 fiducial"
    kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    fid = {"pk": gci.tags_mask_for_sweep(P), "pk_bispec": gci.tags_mask_for_sweep(PB),
           "pk_pgm_bispec": gci.tags_mask_for_sweep(PPB)}
    lists = {
        "kbslice_retrain_rand16": [
            line(PB, ["_kp0.37", "_kb0.24"], 1, 2, kw),
            line(PB, ["_kp0.37", "_kb0.31"], 0, 1, kw),
            line(PPB, ["_kp0.37", "_kpgm0.37", "_kb0.37"], 2, 1, kw),
        ],
        "fresh_pk_rand16": [line(P, fid["pk"], nth, 1, kw, coverage=True) for nth in (1, 2)],
        "fresh_coverage_rand16": (
            [line(PB, fid["pk_bispec"], nth, 1, kw, train=False, ood=False, coverage=True) for nth in (1, 2)]
            + [line(PPB, fid["pk_pgm_bispec"], nth, 1, kw, train=False, ood=False, coverage=True) for nth in (0, 1, 2)]),
    }
    for name, ll in lists.items():
        fn = DIR_LISTS / f"{name}.txt"
        fn.write_text("\n".join(ll) + "\n")
        print(f"wrote {fn} ({len(ll)} lines, {sum(l.startswith('none') for l in ll)} test-only)")


if __name__ == "__main__":
    main()
