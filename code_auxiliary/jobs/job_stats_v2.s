#!/bin/bash
# v2 statistics (per-row on-the-fly noise via tag_noise=_noisefly; P_gm matter = "1" operator).
# Args: statistic tag_params tag_biasparams tag_noise [lib_root] [step_size (cosmologies per task, default 100)]
#   lib_root: where the bias fields live, e.g. /data/kstoreyf/muchisimocks/muchisimocks_lib for training
# Submit from code_auxiliary/ (logs -> /scratch/kstoreyf/muchisimocks/logs_v2/), e.g. training set (array of 100 tasks x 100 cosmologies):
#   sbatch --job-name=pgm_v2_train --array=0-99%40 jobs/job_stats_v2.s pgm _p5_n10000 _biasnoisenestv2_p9_n320000 _noisefly /data/kstoreyf/muchisimocks/muchisimocks_lib
#   bispec (~10 min/cosmology): --time=16:00:00 --array=0-199%40 ... bispec ... /data/kstoreyf/muchisimocks/muchisimocks_lib 50
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G # 3G: pgm peaks ~3.14G, OOM-killed
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%a.out # off /home: home quota blocks kill jobs

statistic=$1
tag_params=$2
tag_biasparams=$3
tag_noise=$4
lib_root=$5
step_size=${6:-100}

echo "Current date and time: $(date)"
echo "Slurm job id ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}, node ${SLURMD_NODENAME}"
echo "statistic=${statistic} tag_params=${tag_params} tag_biasparams=${tag_biasparams} tag_noise=${tag_noise} lib_root=${lib_root}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code
source /home/kstoreyf/muchisimocks/code_auxiliary/env_v2.sh
if [ -n "${lib_root}" ]; then export MUCHISIMOCKS_LIB=${lib_root}; fi
echo "MUCHISIMOCKS_DATA=${MUCHISIMOCKS_DATA} MUCHISIMOCKS_LIB=${MUCHISIMOCKS_LIB}"

idx_mock_start=$((SLURM_ARRAY_TASK_ID*step_size))
idx_mock_end=$((idx_mock_start + step_size))
echo "idx_mock_start=${idx_mock_start}, idx_mock_end=${idx_mock_end}"

python compute_statistics.py --statistic ${statistic} --tag_params ${tag_params} --tag_biasparams ${tag_biasparams} --tag_noise ${tag_noise} --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
