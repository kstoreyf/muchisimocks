#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=datagen_p5_n10000_idx1004
#SBATCH --time=00:30:00
#SBATCH --nodes=1              # nodes per instance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8             # tasks per instance
#SBATCH --mem=35G 	       # 30 hit OOM error
#SBATCH --output=logs/%x.out


echo "Current date and time: $(date)"
echo "Slurm job id is ${SLURM_JOB_ID}"
echo "Running on node ${SLURMD_NODENAME}"
echo "SLURM_CPUS_ON_NODE = ${SLURM_CPUS_ON_NODE}"
echo "SLURM_JOB_NUM_NODES = ${SLURM_JOB_NUM_NODES}"
# https://hpc.nmsu.edu/discovery/slurm/job-arrays/
echo "Array job id is ${SLURM_ARRAY_JOB_ID}" # SLURM_JOB_ID + SLURM_ARRAY_TASK_ID
echo "Instance index is ${SLURM_ARRAY_TASK_ID}."
. ~/load_modules.sh
# via https://stackoverflow.com/a/65183109
#source ~/anaconda3/etc/profile.d/conda.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

idx_LH_start=1004
idx_LH_end=$((idx_LH_start+1))
echo "idx_LH_start=${idx_LH_start}, idx_LH_end=${idx_LH_end}"
python data_creation_pipeline.py ${idx_LH_start} ${idx_LH_end}
#python cuda_minimal.py
#python compute_biased_pks_fields.py
