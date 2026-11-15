#!/bin/bash
# Normalization check of the PolyBin Gaussian B variance (check_bispec_gaussian_var.py).
#   sbatch code_auxiliary/jobs/job_check_bispec_gaussian_var.s
#SBATCH --qos=regular
#SBATCH --job-name=check_bispec_gvar
#SBATCH --time=0:40:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --mem=40G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x.out
echo "Running on node ${SLURMD_NODENAME}, task ${SLURM_ARRAY_TASK_ID}, $(date)"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code_auxiliary
python check_bispec_gaussian_var.py
echo "Done $(date)"
