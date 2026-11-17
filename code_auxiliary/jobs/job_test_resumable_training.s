#!/bin/bash
# Test checkpointed/resumable sweep training end to end (synthetic data, W&B project
# muchisimocks-sbi-test, results under /scratch/kstoreyf/muchisimocks/test_resumable).
#   cd code_auxiliary && sbatch jobs/job_test_resumable_training.s
#SBATCH --qos=regular
#SBATCH --job-name=test_resumable_training
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%j.out

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks-cleanup/code_auxiliary
rm -rf /scratch/kstoreyf/muchisimocks/test_resumable

python test_resumable_training.py stop;   echo "stop exit=$? (expect 3)"
python test_resumable_training.py resume; echo "resume exit=$? (expect 0)"
python test_resumable_training.py full;   echo "full exit=$? (expect 0)"
python test_resumable_training.py check;  echo "check exit=$? (expect 0)"
