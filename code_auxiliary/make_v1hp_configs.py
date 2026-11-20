"""make_v1hp_configs.py

Interim v2 models while the v2 hyperparameter sweeps run: every statistic combo at the v2 fiducial
cuts (generate_config_inference.FIDUCIAL_MASKS), trained on the v2 training set with the top
hyperparameters of the matching v1 sweep (v1 cuts kp0.35/kb0.25/kpgm0.25, v1 data), 3 seeds each.

The v1 sweeps' best_runs.txt and the top trials' config.pkl (hyperparameters only, no posteriors)
are copied into results_v2/results_sbi/sbi<v1 sweep name>/, so best mode always retrains
(sweep_name != the config's own sweep name). Output models: ..._best-rand30v1[_seed{s}].

Tests per model: SHAMe at the 3 densities, phase-matched mock at the 3 densities (individual
realizations + mean), CV set (mean), coverage subset inner500_rand100.

Writes configs + the job list jobs/lists/v1hp_v2_fiducial.txt. Run with env_v2.sh sourced:
  python make_v1hp_configs.py          (seconds; login node is fine)
  sbatch --job-name=train_v1hp_fid --array=0-11 jobs/job_train_v2.s jobs/lists/v1hp_v2_fiducial.txt
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import yaml  # noqa: E402

import generate_config_inference as gci  # noqa: E402

TAG_SWEEP = "-rand30v1"   # names the output models; hyperparameters come from V1_SWEEP
V1_MASKS = {"pk": "_kp0.35", "bispec": "_kb0.25", "pgm": "_kpgm0.25"}
V1_TAG_PARAMSALL = "_p5_n10000_biasnoisenest_p9_n320000_noise_unit_p5_n10000"
SEEDS = (0, 1, 2)
COVERAGE_SUBSET = "inner500_rand100"
FN_LIST = HERE / "jobs" / "lists" / "v1hp_v2_fiducial.txt"
DIR_CODE = HERE.parent / "code"


def v1_sweep_name(statistics):
    masks = "".join(V1_MASKS[s] for s in statistics)
    return f"_muchisimocks_{'_'.join(statistics)}{masks}{V1_TAG_PARAMSALL}_rp_bx32_ntrain10000_sweep-rand30"


def rel(fn):
    return os.path.relpath(fn, DIR_CODE)


def main():
    assert os.environ.get("MUCHISIMOCKS_RESULTS", "").endswith("results_v2"), "source code_auxiliary/env_v2.sh first"
    train_kw = gci.resolve_train_tag_bundle("_p5_n10000", "noisy")
    lines = []
    for statistics in gci.STAT_COMBOS:
        statistics = list(statistics)
        tags_mask = [gci.FIDUCIAL_MASKS[s] for s in statistics]
        sweep_v1 = v1_sweep_name(statistics)
        sweep_dir = Path(os.environ["MUCHISIMOCKS_RESULTS"]) / "results_sbi" / f"sbi{sweep_v1}"
        top = gci.nth_passing_run_id(sweep_dir, 0)
        assert (sweep_dir / top / "config.pkl").is_file() and not (sweep_dir / top / "posterior.p").exists(), sweep_dir
        for seed in SEEDS:
            fn_train = gci.generate_train_config(
                overwrite=True, statistics=statistics, tags_mask=tags_mask, run_mode="best",
                tag_sweep=TAG_SWEEP, seed=seed, **train_kw)
            cfg = yaml.safe_load(open(fn_train))
            cfg["sweep_name"] = sweep_v1          # hyperparameters from the v1 sweep's top trial
            with open(fn_train, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            common = dict(tag_params="_p5_n10000", noise_mode="noisy", overwrite=True,
                          statistics=statistics, tags_mask=tags_mask, tag_sweep=TAG_SWEEP, seed=seed)
            tests = [gci.generate_test_config_from_preset("ood", tag_mock=t, **common) for t in gci.NBAR_TAGS_SHAME]
            for nbar_tag in gci.NBAR_TAGS_SHAME:
                nbar = nbar_tag[len("_nbar"):]
                for mean in (False, True):
                    tests.append(gci.generate_test_config(
                        overwrite=True, statistics=statistics, tags_mask=tags_mask, tag_sweep=TAG_SWEEP,
                        seed=seed, **train_kw, evaluate_mean=mean,
                        tag_params_test=f"_shamematched{nbar}_p0_n10",
                        tag_biasparams_test=f"_biasshame{nbar}_noisebest_p0_n1",
                        tag_noise_test="_noisematched"))
            tests.append(gci.generate_test_config_from_preset("fixed_cosmo_shame_mean", **common))
            tests.append(gci.generate_test_config_from_preset("coverage", idxs_obs_subset=COVERAGE_SUBSET, **common))
            for fn in [fn_train] + tests:
                assert Path(fn).is_file(), fn
            lines.append(" ".join(rel(f) for f in [fn_train] + tests))
    FN_LIST.write_text("\n".join(lines) + "\n")
    print(f"wrote {FN_LIST} ({len(lines)} models, {len(lines[0].split()) - 1} tests each)")


if __name__ == "__main__":
    main()
