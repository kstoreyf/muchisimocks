#!/bin/bash
# v2 best-run selection for a finished sweep (rank by validation loss + coverage stall test).
# Arg: statistics tag, e.g. _pk_pgm. Extra args are passed to choose_best_run.py.
#   cd code_auxiliary && sbatch --job-name=choosebest_v2_pk_pgm jobs/job_choose_best_v2.s _pk_pgm
# Always --overwrite: without it, picks from a mid-sweep run are kept even if later trials rank higher.
#SBATCH --qos=regular
#SBATCH --job-name=choosebest_v2
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%j.out

tags_stat=$1
shift
echo "$(date) | job ${SLURM_JOB_ID} node ${SLURMD_NODENAME} | tags_stat=${tags_stat}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code
source /home/kstoreyf/muchisimocks/code_auxiliary/env_v2.sh

python choose_best_run.py --tags_stat "${tags_stat}" --noise-modes noisy --n-passing 3 --overwrite "$@" || exit 1
echo "Done $(date)"
