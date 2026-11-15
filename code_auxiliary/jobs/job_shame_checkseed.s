#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=shame_checkseed
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --mem=20G
#SBATCH --output=logs/%x.out
echo "Running on node ${SLURMD_NODENAME}, $(date)"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code_auxiliary
# -406168: candidate SHAMe/Rings seed; 0 as a wrong-seed control
python make_shame_matched_mock.py check_seed --seeds -406168 0 --phase 0
python make_shame_matched_mock.py check_seed --seeds -406168 --phase pi
