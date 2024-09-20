#!/bin/bash
#SBATCH --qos=regular
##SBATCH --job-name=compute_pks_p5_n10000
##SBATCH --job-name=timetest_datagen_nthreads8
#SBATCH --job-name=id_bad_idxs
#SBATCH --time=12:00:00
#SBATCH --nodes=1              # nodes per instance
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16             # tasks per instance
#SBATCH --mem=35G 	       # 
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
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
#python compute_biased_pks_fields.py
#python data_creation_pipeline.py 4
python identify_bad_mocks.py

#python cuda_minimal.py
