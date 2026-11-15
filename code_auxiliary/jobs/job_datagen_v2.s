#!/bin/bash
# v2 test-set field generation (independent IC phases; seed bases in data_creation_pipeline.SEED_BASE_IC).
# Submit from code_auxiliary/ (logs -> /scratch/kstoreyf/muchisimocks/logs_v2/):
#   sbatch --job-name=datagen_coveragev2_p5_n1000 jobs/job_datagen_v2.s lh _coveragev2_p5_n1000
#   sbatch --job-name=datagen_shamev2_p0_n1000   jobs/job_datagen_v2.s fixed _shamev2_p0_n1000
# 10 tasks x 100 mocks, 1 GPU each (same layout as jobarr_datagen.s).
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --mem=35G # got OOM for 30 for datagen
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%a.out # off /home: home quota blocks kill jobs

modecosmo=$1
tag_params=$2

echo "Current date and time: $(date)"
echo "Slurm job id is ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}, node ${SLURMD_NODENAME}"
echo "modecosmo=${modecosmo}, tag_params=${tag_params}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code
source /home/kstoreyf/muchisimocks/code_auxiliary/env_v2.sh

step_size=100
idx_mock_start=$((SLURM_ARRAY_TASK_ID*step_size))
idx_mock_end=$((idx_mock_start + step_size))
echo "idx_mock_start=${idx_mock_start}, idx_mock_end=${idx_mock_end}"

python data_creation_pipeline.py ${idx_mock_start} ${idx_mock_end} --modecosmo ${modecosmo} --tag_params ${tag_params}
