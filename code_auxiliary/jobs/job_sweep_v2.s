#!/bin/bash
# v2 W&B hyperparameter sweep: one array task = one sweep agent = one trial, all on the same sweep.
# Arg: a run_mode=sweep training config (paths relative to code/), from generate_config_inference.main().
#   cd code_auxiliary && sbatch --array=0-15 --job-name=sweep_v2_pk_pgm jobs/job_sweep_v2.s \
#       ../configs/configs_train/config_muchisimocks_pk_pgm_kp0.37_kpgm0.24_..._sweep-rand16.yaml
# --array size = sweep_num_runs (SWEEP_NUM_RUNS=16). The first task creates the sweep and writes
# wandb_sweep_id into the config; the others wait on the sweep lock and join it.
# A trial killed at the wall time is logged to <sweep dir>/killed_trials.csv (its run dir is kept
# under killed_trials/) by the next agent that starts, and the sweep then draws a replacement:
# resubmit as many tasks as there are missing trials (an agent exits at once if the sweep is full).
#SBATCH --qos=long
#SBATCH --job-name=sweep_v2
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1   # training is single-threaded (bench_train_threads.py: more threads are slower)
#SBATCH --mem=30G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%A_%a.out

config_train=$1
echo "$(date) | job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} node ${SLURMD_NODENAME}"
echo "sweep config: ${config_train}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code
source /home/kstoreyf/muchisimocks/code_auxiliary/env_v2.sh

export MUCHISIMOCKS_SWEEP_PARALLEL=1        # agents share the sweep; skip orphan-dir cleanup
export MUCHISIMOCKS_SWEEP_RUNS_PER_AGENT=1  # one trial per task, so each trial gets the full wall time
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 5 ))  # stagger W&B/lock traffic

python run_inference.py --config-train ${config_train} || exit 1
echo "Done $(date)"
