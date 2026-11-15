#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=shame_matched_noisepoly
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --mem=40G
#SBATCH --output=logs/%x.out
echo "Running on node ${SLURMD_NODENAME}, $(date)"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code_auxiliary
python compute_shame_matched_noisepoly.py
echo "Done $(date)"
