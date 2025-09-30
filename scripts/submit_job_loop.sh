#!/bin/bash

#n_train_arr=(500)
#n_train_arr=(500 1000 2000 4000 6000 8000 10000)
n_train_arr=(10000)
#n_train_arr=(500 1000 2000 4000 6000 8000)
#tag_stats_arr=("_pk" "_bispec" "_pk_bispec") 
#tag_stats_arr=("_bispec" "_pk_bispec") 
#tag_stats_arr=("_pk") 
tag_stats_arr=("_bispec") 
#tag_stats_arr=("_pk" "_bispec") 
#tag_stats_arr=("_pk_bispec")
#tag_stats_arr=("_bispec" "_pk_bispec") 

#n_train_arr=(10000)
#tag_stats_arr=("_pk_bispec")#

for n_train in "${n_train_arr[@]}"; do
    for tag_stats in "${tag_stats_arr[@]}"; do
        ### TRAINING ###
        tag_params="_p5_n10000"
        tag_biasparams="_biaszen_p4_n200000"  
        tag_noise="_noise_p5_n10000"
        #tag_Anoise="_An_p1_n10000"    
        #tag_Anoise="_An1_p0_n1"    
        ## no noise
        tag_noise=""
        tag_Anoise=""   
        #config_train_file="../configs/configs_train/config_muchisimocks${tag_stats}${tag_params}${tag_biasparams}${tag_noise}${tag_Anoise}_ntrain${n_train}.yaml"
        # if only want to train on a pre-trained model, set config_train_file to "none";
        # but if you accidentally leave config_train_file not blank, the default is not to overwrite, so it shouldn't matter!
        config_train_file="none"

        ### TESTING ###
        ### cosmic variance (quijote)
        # tag_params_test="_quijote_p0_n1000"
        # tag_biasparams_test="_b1000_p0_n1"
        #tag_mean="_mean"
        #tag_noise_test="_noise_quijote_p0_n1000"
        #tag_Anoise_test="_An1_p0_n1"
        ### coverage
        tag_params_test="_test_p5_n1000"
        tag_biasparams_test="_biaszen_p4_n1000"
        tag_mean=""
        # tag_noise_test="_noise_test_p5_n1000"
        # tag_Anoise_test="_An_p1_n1000"
        ### no noise
        tag_noise_test=""
        tag_Anoise_test=""

        ### Muchisimocks test set 
        config_test_file="../configs/configs_test/config_TRAIN_muchisimocks${tag_stats}${tag_params}${tag_biasparams}${tag_noise}${tag_Anoise}_ntrain${n_train}_TEST_muchisimocks${tag_stats}${tag_params_test}${tag_biasparams_test}${tag_noise_test}${tag_Anoise_test}${tag_mean}.yaml"
        ### OOD test set
        # data_mode="shame"
        # tag_mock="_An1"
        # config_test_file="../configs/configs_test/config_TRAIN_muchisimocks${tag_stats}${tag_params}${tag_biasparams}${tag_noise}${tag_Anoise}_ntrain${n_train}_TEST_${data_mode}${tag_stats}${tag_mock}.yaml"
        ### no test
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
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}.out
##SBATCH --time=24:00:00 #24h for testing on coverage test set (24h is max time limit; some dont converge)
##SBATCH --time=1:00:00 #1h for testing on cosmic var test set or single OOD
#SBATCH --time=4:00:00 #2h for training -> 4 in case
##SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G #40gb for both training and testing (30 failed one time)

echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Running on node \${SLURMD_NODENAME}"

echo "config_train_file: ${config_train_file}"
echo "config_test_file: ${config_test_file}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

echo "python run_inference.py \\
    $( [[ "$config_train_file" != "none" ]] && echo "--config-train=${config_train_file}" ) \\
    $( [[ "$config_test_file" != "none" ]] && echo "--config-test=${config_test_file}" )"

python run_inference.py \\
    $( [[ "$config_train_file" != "none" ]] && echo "--config-train=${config_train_file}" ) \\
    $( [[ "$config_test_file" != "none" ]] && echo "--config-test=${config_test_file}" )
EOF

    done
done