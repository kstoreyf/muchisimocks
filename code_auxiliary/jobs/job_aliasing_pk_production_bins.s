#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=alias_pk
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x.out

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code
python ../code_auxiliary/aliasing_pk_production_bins.py
