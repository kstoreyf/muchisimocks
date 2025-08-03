#!/bin/bash
#SBATCH --qos=regular
##SBATCH --job-name=datagen_p5_n10000_idx1004
#SBATCH --job-name=gen_noise_fields
##SBATCH --job-name=run_inf_sbi_p5_n10000_b1000_p0_n1_quijote_p0_n1000_b1000_p0_n1_samp10000
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n10000_quijote_p0_n1000_biaszen_p4_n1000_nsf_samp10000
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n10000_quijote_p0_n1000_b1000_p0_n1_nsf
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n100000_quijote_p0_n1000_b1000_p0_n1
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n100000_test_p5_n1000_biaszen_p4_n1000
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n100000_ntrain100000_sweep-sbi-rand10
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n100000_ntrain6000_best-sbi-rand10
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n100000_ntrain500_best-sbi-rand10_test_p5_n1000_biaszen_p4_n1000
##SBATCH --job-name=run_inf_sbi_p5_n10000_biaszen_p4_n100000_ntrain600000_test_p5_n1000_biaszen_p4_n1000
##SBATCH --job-name=run_inf_sbi_p5_n10000_b1zen_n10000_test_p5_n1000_b1zen_p1_n1000
##SBATCH --job-name=run_inf_sbi_p5_n10000_b1000_p0_n1_test_p5_n1000_b1000_p0_n1
##SBATCH --job-name=run_inf_sbi_p5_n10000_b1zen_p1_n10000_best-sbi-rand10
##SBATCH --job-name=run_inf_TRAIN_muchisimocksPk_p5_n10000_biaszen_p4_n100000_ntrain1000_best-sbi-rand10_TEST_muchisimocksPk_test_p5_n1000_biaszen_p4_n1000
##SBATCH --job-name=run_inf_TRAIN_muchisimocks_bispec_p5_n10000_biaszen_p4_n100000_ntrain10000_TEST_muchisimocks_bispec_test_p5_n1000_biaszen_p4_n1000
##SBATCH --job-name=run_inf_train_muchisimocks_pk_p5_n10000_biaszen_p4_n100000_ntrain10000_best-sbi-rand10
##SBATCH --job-name=run_inf_train_muchisimocks_bispec_p5_n10000_biaszen_p4_n10000_ntrain10000
##SBATCH --job-name=run_inf_emu_pk_p5_n10000_biaszen_p4_n10000_boxsize1000_nrlzs1_ntrain10000
#SBATCH --time=4:00:00
#SBATCH --nodes=1              # nodes per instance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=24             # tasks per instance
#SBATCH --mem=3G 	       # 35 for datagen (30 hit oom)
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

#python cuda_minimal.py
#python compute_biased_pks_fields.py
python generate_noise_fields.py
