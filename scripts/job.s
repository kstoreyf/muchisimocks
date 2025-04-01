#!/bin/bash
#SBATCH --qos=regular
##SBATCH --job-name=datagen_p5_n10000_idx1004
##SBATCH --job-name=run_inf_sbi_p5_n10000_b1000_p0_n1_quijote_p0_n1000_b1000_p0_n1_samp10000
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n10000_quijote_p0_n1000_biaszen_p4_n1000_nsf_samp10000
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n10000_quijote_p0_n1000_b1000_p0_n1_nsf
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n100000_quijote_p0_n1000_b1000_p0_n1
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n10000_test_p5_n1000_biaszen_p4_n1000
##SBATCH --job-name=run_inf_sbi_p5_n10000_b1zen_n10000_test_p5_n1000_b1zen_p1_n1000
#SBATCH --job-name=run_inf_sbi_p5_n10000_b1000_p0_n1_test_p5_n1000_b1000_p0_n1
##SBATCH --job-name=run_inf_sbi_p5_n10000_b1zen_p1_n10000_best-sbi-rand10
#SBATCH --time=24:00:00
#SBATCH --nodes=1              # nodes per instance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8             # tasks per instance
#SBATCH --mem=30G 	       # 35 for datagen (30 hit oom)
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

python run_inference.py

#idx_LH_start=1004
#idx_LH_end=$((idx_LH_start+1))
#echo "idx_LH_start=${idx_LH_start}, idx_LH_end=${idx_LH_end}"
#python data_creation_pipeline.py ${idx_LH_start} ${idx_LH_end}

#python cuda_minimal.py
#python compute_biased_pks_fields.py
