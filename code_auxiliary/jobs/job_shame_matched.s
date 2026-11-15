#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=shame_matched_mock
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --mem=40G
#SBATCH --output=logs/%x.out
echo "Running on node ${SLURMD_NODENAME}, $(date)"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code_auxiliary
# seed -406168 verified via check_seed: r(k<0.05)=+0.99 vs SHAMe dens mesh, both phases, phase_type=1
python make_shame_matched_mock.py make --seed -406168 --phases 0 pi
echo "Done $(date)"
