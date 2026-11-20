#!/bin/bash
# Benchmark posterior sampling speed per CPU thread count (+ GPU if requested with --gres).
#   sbatch --job-name=bench_sample_cpu jobs/job_bench_sample.s <dir_sbi>
#   sbatch --job-name=bench_sample_gpu --gres=gpu:1 --cpus-per-task=4 jobs/job_bench_sample.s <dir_sbi> --threads 1 4
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%j.out
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code_auxiliary
source ./env_v2.sh
python ${BENCH:-bench_sample_threads.py} "$@"
