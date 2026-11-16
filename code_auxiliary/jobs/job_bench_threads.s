#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=bench_threads
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x.out
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code_auxiliary
lscpu | grep -E "Model name|^CPU\(s\)"
python bench_train_threads.py --threads 1 --bs 64 128 --hidden 512 256 128
