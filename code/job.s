#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=xx
#SBATCH --time=4:00:00
#SBATCH --nodes=1              # nodes per instance
#SBATCH --gres=gpu:1 		   #gpu only needed for datagen!
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=24             # tasks per instance
##SBATCH --ntasks=4             # tasks per instance
#SBATCH --mem=5G 	       
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

#python compute_run_ess.py --noise-modes noiseless --stat-labels pk_bispec

#python run_inference.py

#python run_inference.py \
#	--tr=../configs/configs_train/config_muchisimocks_pk_p5_n10000_biaszen_p4_n100000_ntrain10000_best-sbi-rand10.yaml
	#--tr=../configs/configs_train/config_muchisimocks_bispec_p5_n10000_biaszen_p4_n100000_ntrain10000.yaml \
	#--t=../configs/configs_test/config_TRAIN_muchisimocks_bispec_p5_n10000_biaszen_p4_n100000_ntrain10000_TEST_muchisimocks_bispec_test_p5_n1000_biaszen_p4_n1000.yaml \
	#--tr=../configs/configs_train/config_emu_pk_p5_n10000_biaszen_p4_n10000_boxsize1000_nrlzs1_ntrain10000.yaml
	#--config-train=../configs/configs_train/#config_muchisimocks_bispec_p5_n10000_biaszen_p4_n10000_ntrain10000.yaml \
	#--config-test=../configs/configs_test/config_TRAIN_muchisimocks_bispec_p5_n10000_biaszen_p4_n10000_ntrain10000_TEST_muchisimocks_bispec_quijote_p0_n1000_b1000_p0_n1_mean.yaml
	#--config-train=../configs/configs_train/config_muchisimocksPk_p5_n10000_biaszen_p4_n100000_ntrain1000_best-sbi-rand10.yaml \
	#--config-test=../configs/configs_test/config_TRAIN_muchisimocksPk_p5_n10000_biaszen_p4_n100000_ntrain1000_best-sbi-rand10_TEST_muchisimocksPk_test_p5_n1000_biaszen_p4_n1000.yaml
	#--config-test=../configs/configs_test/config_TRAIN_muchisimocks_bispec_p5_n10000_biaszen_p4_n10000_ntrain10000_TEST_muchisimocks_bispec_quijote_p0_n1000_b1000_p0_n1_mean.yaml

#idx_LH_start=1004
#idx_LH_end=$((idx_LH_start+1))
#echo "idx_LH_start=${idx_LH_start}, idx_LH_end=${idx_LH_end}"
#python data_creation_pipeline.py ${idx_LH_start} ${idx_LH_end}

#python data_creation_pipeline.py
#python cuda_minimal.py
#python compute_biased_pks_fields.py
#python generate_noise_fields.py
