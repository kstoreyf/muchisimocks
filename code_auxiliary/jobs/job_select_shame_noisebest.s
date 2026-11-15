#!/bin/bash
# B/P emulator coefficients for select_shame_noisebest.py: one task per (phase, eps realization).
# Then: sbatch --dependency=afterok:<this job> code_auxiliary/jobs/job_select_shame_noisebest_fit.s
#   sbatch code_auxiliary/jobs/job_select_shame_noisebest.s
#SBATCH --qos=regular
#SBATCH --job-name=select_noisebest
#SBATCH --array=0-7
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --mem=40G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%a.out
echo "Running on node ${SLURMD_NODENAME}, task ${SLURM_ARRAY_TASK_ID}, $(date)"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code_auxiliary
python select_shame_noisebest.py emulate ${SLURM_ARRAY_TASK_ID}
echo "Done $(date)"
