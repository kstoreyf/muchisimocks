"""make_ppb_ext_transfer_rand16.py

Optimistic transfer test for the P+P_gm+B ext2/ext3 grid trials (2026-11-24, user OK'd): train each trial's
hyperparameters at kb0.27 and kb0.37 (kp0.37 / kpgm0.37, seed 0) before knowing whether it beats setting 3
(bydsd9rq, -2.425) at the sweep cut kb0.24, so the 0.4-nat transfer check is ready as soon as the sweep finishes.
Training only (no tests): the decision uses best val loss vs the best member at each kb (11-19 nb B3).

Hyperparameters come from the trial's config.pkl via sweep_run_id (sbi_model run_mode best), which only exists
once the trial has finished, so each line must run with --dependency=afterok:<that trial's sweep task>.
Model names: ..._best-rand16-topkpgm0.37_trial{run_id}. If a trial becomes setting 3, rerun choose_best and
rename/copy these dirs to the _nbest{n} name (or retrain under it) so the paper figs pick them up.

Run with env_v2.sh sourced (seconds; login node is fine):
  python make_ppb_ext_transfer_rand16.py
  then sbatch per trial (see the printed commands).
"""
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DIR_CFG = HERE.parent / "configs" / "configs_train"
FN_LIST = HERE / "jobs" / "lists" / "ppb_ext_transfer_rand16.txt"
BASE = "config_muchisimocks_pk_pgm_bispec_kp0.37_kpgm0.37_kb{kb}_p5_n10000_biasnoisenestv2_p9_n320000_noisefly_rp_bx32_ntrain10000_best-rand16-topkpgm0.37"
# (W&B run id, sweep job array task it comes from)
TRIALS = [("bss3nsz9", "8175128_0"),   # ext2: 512 / 1e-4 / 10 transforms / batch 64
          ("xi4em9br", "8175129_0"),   # ext3: 512 / 1e-4 / 8 transforms / batch 128 or 256
          ("zr8o6ys9", "8175129_1")]   # ext3: the other batch size
KBS = ("0.27", "0.37")


def main():
    assert os.environ.get("MUCHISIMOCKS_DATA", "").endswith("data_v2"), "source code_auxiliary/env_v2.sh first"
    lines, cmds = [], []
    for i, (run_id, task) in enumerate(TRIALS):
        for kb in KBS:
            template = DIR_CFG / (BASE.format(kb=kb) + "_nbest3.yaml")
            cfg = yaml.safe_load(template.read_text())
            cfg.pop("nth_best_run")
            cfg["sweep_run_id"] = run_id
            cfg["tag_inf"] = cfg["tag_inf"].replace("_nbest3", f"_trial{run_id}")
            assert cfg["tag_inf"].endswith(f"topkpgm0.37_trial{run_id}"), cfg["tag_inf"]
            fn = DIR_CFG / (BASE.format(kb=kb) + f"_trial{run_id}.yaml")
            fn.write_text(yaml.safe_dump(cfg, sort_keys=True))
            lines.append(f"../configs/configs_train/{fn.name}")
        a = 2 * i
        cmds.append(f"sbatch --job-name=train_ppb_ext_transfer_{run_id} --array={a}-{a + 1} "
                    f"--dependency=afterok:{task} jobs/job_train_v2.s jobs/lists/{FN_LIST.name}")
    FN_LIST.write_text("\n".join(lines) + "\n")
    print(f"wrote {FN_LIST} ({len(lines)} lines)")
    print("\n".join(cmds))


if __name__ == "__main__":
    main()
