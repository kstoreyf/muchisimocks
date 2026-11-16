#!/bin/bash
# v2 SBI training + SHAMe OOD tests, one array task per model.
# List file: one line per model, "<train config> <test config> [<test config> ...]" (paths relative to code/);
# train config "none" = test only (model already trained).
#   sbatch --array=0-5 jobs/job_train_v2.s jobs/lists/noise_ab_v2.txt      (submit from code_auxiliary/)
#SBATCH --qos=regular
#SBATCH --job-name=train_v2
#SBATCH --time=8:00:00
#SBATCH --nodes=1
##SBATCH --gres=gpu:1  # not needed for training
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1   # training is single-threaded (bench_train_threads.py: more threads are slower)
#SBATCH --mem=30G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%A_%a.out

list=$(realpath $1)
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${list})
read -r config_train configs_test <<< "${line}"
echo "$(date) | job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} node ${SLURMD_NODENAME}"
echo "train: ${config_train}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code
source /home/kstoreyf/muchisimocks/code_auxiliary/env_v2.sh

if [ "${config_train}" != "none" ]; then
    python run_inference.py --config-train ${config_train} || exit 1
fi
for config_test in ${configs_test}; do
    echo "test: ${config_test}"
    python run_inference.py --config-test ${config_test} || exit 1
done
echo "Done $(date)"
