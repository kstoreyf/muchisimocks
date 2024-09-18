#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=datagen_p5_n10000_step7_round2
##SBATCH --job-name=cuda_testgpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1              # nodes per instance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8             # tasks per instance
##x-y%z; start x, end y INCLUSIVE, z tasks at a time max
#SBATCH --array=3000-3999%5
#SBATCH --mem=35G 	       # 30 hit OOM error
##SBATCH --output=logs/%x.out
#SBATCH --output=logs/%x-%a.out

##SBATCH --output=logs/%x-%j.out
##SBATCH --error=%x-%j.err

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
i=$((SLURM_ARRAY_TASK_ID-SLURM_ARRAY_TASK_MIN))
step_size=7
echo "i=${i}"
idx_LH_start=$((SLURM_ARRAY_TASK_MIN + i*step_size))
idx_LH_end=$((idx_LH_start + step_size))
echo "idx_LH_start=${idx_LH_start}, idx_LH_end=${idx_LH_end}"
python data_creation_pipeline.py ${idx_LH_start} ${idx_LH_end}
#python cuda_minimal.py
