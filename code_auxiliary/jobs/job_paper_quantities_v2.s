#!/bin/bash
# v2 paper quantities -> figures/paper_quantities.dat (v1: paper_quantities_v1.dat) (partial ensembles + XX placeholders, see the script header).
#   cd code_auxiliary && sbatch --job-name=paperquant_v2 jobs/job_paper_quantities_v2.s
#SBATCH --qos=regular
#SBATCH --job-name=paperquant_v2
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%j.out

echo "$(date) | job ${SLURM_JOB_ID} node ${SLURMD_NODENAME}"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
source /home/kstoreyf/muchisimocks/code_auxiliary/env_v2.sh
cd /home/kstoreyf/muchisimocks
python code/compute_paper_quantities.py "$@" || exit 1
echo "Done $(date)"
