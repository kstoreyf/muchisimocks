#!/bin/bash

#n_train_arr=(500)
#n_train_arr=(500 1000 2000 4000 6000 8000 10000)
n_train_arr=(10000)
#n_train_arr=(500 1000 2000 4000 6000 8000)
#tag_stats_arr=("_pk" "_bispec" "_pk_bispec") 
#tag_stats_arr=("_pk" "_bispec") 
tag_stats_arr=("_pk")

#n_train_arr=(10000)
#tag_stats_arr=("_pk_bispec")

for n_train in "${n_train_arr[@]}"; do
    for tag_stats in "${tag_stats_arr[@]}"; do
        tag_params="_p5_n10000"
        #tag_biasparams="_biaszen_p4_n50000"  
        tag_biasparams="_biaszen_p4_n200000"  
        #config_train_file="../configs/configs_train/config_muchisimocks${tag_stats}${tag_params}${tag_biasparams}_ntrain${n_train}.yaml"
        config_train_file="none"

        tag_noise_test="_noise_quijote_p0_n1000"
        tag_Anoise_test="_An1_p0_n1"
        #config_test_file="../configs/configs_test/config_TRAIN_muchisimocks${tag_stats}${tag_params}${tag_biasparams}_ntrain${n_train}_TEST_muchisimocks${tag_stats}_test_p5_n1000_biaszen_p4_n1000.yaml"
        config_test_file="../configs/configs_test/config_TRAIN_muchisimocks${tag_stats}${tag_params}${tag_biasparams}_ntrain${n_train}_TEST_muchisimocks${tag_stats}_quijote_p0_n1000_b1000_p0_n1${tag_noise_test}${tag_Anoise_test}_mean.yaml"
        #config_test_file="none"

        # Determine job name logic
        if [[ "$config_train_file" == "none" && "$config_test_file" == "none" ]]; then
            echo "Both config_train and config_test are none, skipping job."
            continue
        elif [[ "$config_train_file" == "none" ]]; then
            config_base=$(basename "$config_test_file" .yaml)
            job_name="run_inf_${config_base}"
        elif [[ "$config_test_file" == "none" ]]; then
            config_base=$(basename "$config_train_file" .yaml)
            job_name="run_inf_${config_base}"
        else
            config_base=$(basename "$config_test_file" .yaml)
            job_name="run_inf_traintest_${config_base}"
        fi

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}.out
#SBATCH --time=24:00:00 #24h for testing on coverage test set (24h is max time limit; some dont converge)
##SBATCH --time=1:00:00 #1h for testing on cosmic var test set
##SBATCH --time=2:00:00 #2h for training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G #30gb for both training and testing (actually failed once, now doing 40)

echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Running on node \${SLURMD_NODENAME}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

python run_inference.py \\
    $( [[ "$config_train_file" != "none" ]] && echo "--config-train=${config_train_file}" ) \\
    $( [[ "$config_test_file" != "none" ]] && echo "--config-test=${config_test_file}" )
EOF

    done
done