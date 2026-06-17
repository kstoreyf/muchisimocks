#!/bin/bash
# Submit: cd "$(dirname "$0")" && mkdir -p logs && sbatch "$0"
#SBATCH --qos=regular
#SBATCH --job-name=matched_mocks_0663_damping
##SBATCH --job-name=generate_emupks_p5_n10000_biaszen_p4_n1000
##SBATCH --job-name=run_inf_train_muchisimocks_pk_bispec_p5_n10000_biaszen_p4_n100000_ntrain10000_sweep-rand10
#SBATCH --time=1:00:00 # max 24h
#SBATCH --nodes=1               # nodes per instance
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1    	# leave this 1
#SBATCH --ntasks=24             # tasks per instance - match nthreads here
#SBATCH --mem=35G 	       # 
##SBATCH --mem=50G #need >15 for compute_pks # 45 hit OOM for pnn
##SBATCH --mem=30G # 30 for training
##SBATCH --mem=60G # 
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

# 0037 0574 0822 1082 1510 0254 0663 0977 1317 1642
python make_quijote_matched_mocks.py --idxs_LH 0663 --include_damping


# old way was extremely picky with the backslashes so doing this! claude's idea
# args=(
# 	#--config-train=../configs/configs_train/config_muchisimocks_bispec_p5_n10000_biaszen_p4_n100000_ntrain1000.yaml
# 	#--config-train=../configs/configs_train/config_muchisimocks_bispec_p5_n10000_biaszen_p4_n100000_ntrain10000.yaml
# 	#--config-train=../configs/configs_train/config_muchisimocks_pk_bispec_p5_n10000_biaszen_p4_n100000_ntrain1000_best-rand10.yaml
# 	#--config-test=../configs/configs_test/config_TRAIN_muchisimocks_pk_bispec_p5_n10000_biaszen_p4_n100000_ntrain1000_TEST_muchisimocks_pk_bispec_test_p5_n1000_biaszen_p4_n1000.yaml
# 	#--config-train=../configs/configs_train/config_muchisimocks_pk_p5_n10000_biaszen_p4_n100000_ntrain10000_best-rand10.yaml
# 	#--config-test=../configs/configs_test/config_TRAIN_muchisimocks_pk_p5_n10000_biaszen_p4_n100000_ntrain10000_best-rand10_TEST_muchisimocks_pk_test_p5_n1000_biaszen_p4_n1000.yaml
#     #--config-train=../configs/configs_train/config_muchisimocks_bispec_p5_n10000_biaszen_p4_n100000_ntrain10000.yaml
#     --config-test=../configs/configs_test/config_TRAIN_muchisimocks_bispec_p5_n10000_biaszen_p4_n100000_ntrain500_TEST_muchisimocks_bispec_test_p5_n1000_biaszen_p4_n1000.yaml
# )
# python run_inference.py "${args[@]}"


#python run_inference.py \
	#--config-train=../configs/configs_train/config_muchisimocks_pk_bispec_p5_n10000_biaszen_p4_n100000_ntrain10000_sweep-rand10.yaml
	#--config-train=../configs/configs_train/config_muchisimocks_pk_p5_n10000_biaszen_p4_n100000_ntrain10000.yaml \
	#--config-test=../configs/configs_test/config_TRAIN_muchisimocks_pk_bispec_p5_n10000_biaszen_p4_n100000_ntrain10000_TEST_muchisimocks_pk_bispec_test_p5_n1000_biaszen_p4_n1000.yaml
	#--config-test=../configs/configs_test/config_TRAIN_muchisimocks_pk_bispec_p5_n10000_biaszen_p4_n10000_ntrain10000_TEST_muchisimocks_pk_bispec_test_p5_n1000_biaszen_p4_n1000.yaml

#python compute_pnns.py
#python run_inference.py
#python generate_emuPks.py
#python compute_biased_pks_fields.py
#python data_creation_pipeline.py 4
#python identify_bad_mocks.py
#python compute_theoretical_quantities.py

