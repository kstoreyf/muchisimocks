#!/bin/bash
#SBATCH --qos=regular
##SBATCH --job-name=precompute_kaiser_p5_n10000_b1
##SBATCH --job-name=compute_pnn_emu_p5_n10000
##SBATCH --job-name=compute_pks_p5_n10000_b0000_cont
#SBATCH --job-name=compute_pks_p5_n10000_biaszen_p1_n10000_cont
##SBATCH --job-name=emcee_emuPk_5param_n10000
##SBATCH --job-name=timetest_datagen_nthreads8
##SBATCH --job-name=id_bad_idxs
#SBATCH --time=24:00:00
#SBATCH --nodes=1              # nodes per instance
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=24             # tasks per instance
##SBATCH --mem=35G 	       # 
#SBATCH --mem=25G #need >15 for compute_pks
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
#python run_inference.py
python compute_biased_pks_fields.py
#python data_creation_pipeline.py 4
#python identify_bad_mocks.py
#python compute_theoretical_quantities.py

#python cuda_minimal.py
