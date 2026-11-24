# Source this in every job of the v2 rerun (independent test-set phases, per-draw noise,
# P_gm/covariance/validation fixes). Outputs land in parallel *_v2 trees; old outputs untouched.
# New test-set mock libraries (_coveragev2_p5_n1000, _cvshamev2_p0_n1000) are written to the
# default scratch lib root (/scratch/kstoreyf/muchisimocks/muchisimocks_lib<tag>).
# NOTE: the training fields (_p5_n10000) live in /data/kstoreyf/muchisimocks/; set
# MUCHISIMOCKS_LIB=/data/kstoreyf/muchisimocks/muchisimocks_lib when computing training stats.
export MUCHISIMOCKS_DATA=/scratch/kstoreyf/muchisimocks/data_v2
export MUCHISIMOCKS_RESULTS=/scratch/kstoreyf/muchisimocks/results_v2
# W&B's local run dirs (wandb/run-*) go to scratch, not code/wandb in home: they filled the home quota on
# 2026-11-23 (~08:00, EDQUOT killed 14 training tasks). Runs still sync to wandb.ai as before.
export WANDB_DIR=/scratch/kstoreyf/muchisimocks/wandb_v2
mkdir -p "${WANDB_DIR}"
