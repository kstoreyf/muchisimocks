#!/bin/bash
# Fit stage of select_shame_noisebest.py (needs the emulate array, job_select_shame_noisebest.s).
#   sbatch code_auxiliary/jobs/job_select_shame_noisebest_fit.s
#SBATCH --qos=regular
#SBATCH --job-name=select_noisebest_fit
#SBATCH --time=0:30:00
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
python select_shame_noisebest.py fit
echo "Done $(date)"
