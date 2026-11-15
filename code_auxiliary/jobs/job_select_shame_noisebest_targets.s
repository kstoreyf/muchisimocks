#!/bin/bash
# SHAMe targets + Gaussian errors (3 densities) for select_shame_noisebest.py.
#   sbatch code_auxiliary/jobs/job_select_shame_noisebest_targets.s
#SBATCH --qos=regular
#SBATCH --job-name=select_noisebest_targets
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
python select_shame_noisebest.py targets
echo "Done $(date)"
