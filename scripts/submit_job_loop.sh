#!/bin/bash

#n_train_arr=(500)
#n_train_arr=(500 1000 2000 4000 6000 8000 10000)
n_train_arr=(10000)
#n_train_arr=(500 1000 2000 4000 6000 8000)
tag_stats_arr=("_pk" "_bispec" "_pk_bispec") 
#tag_stats_arr=("_pk" "_bispec") 
#tag_stats_arr=("_bispec" "_pk_bispec") 
#tag_stats_arr=("_pk") 
#tag_stats_arr=("_bispec") 
#tag_stats_arr=("_pk_bispec")

for n_train in "${n_train_arr[@]}"; do
    for tag_stats in "${tag_stats_arr[@]}"; do
        ### TRAINING ###
        tag_params="_p5_n10000"
        #tag_biasparams="_biaszen_p4_n10000"  
        tag_biasparams="_biaszen_p4_n200000"  
        # tag_noise="_noise_unit_p5_n10000"
        # tag_Anoise="_Anmult_p5_n200000"
        tag_noise="_noise_unit_p5_n10000"
        tag_Anoise="_Anmult_p5_n200000"
        #tag_Anoise="_An_p1_n10000" # free Anoise
        #tag_Anoise="_An1_p0_n1" # fix Anoise=1
        ## no noise
        # tag_noise=""
        # tag_Anoise=""   
        #tag_mask=""
        #tag_mask="_kmaxbispec0.25"
        tag_mask=""
        tag_data_train="muchisimocks${tag_stats}${tag_mask}${tag_params}${tag_biasparams}${tag_noise}${tag_Anoise}_ntrain${n_train}"
        #config_train_file="../configs/configs_train/config_${tag_data_train}.yaml"
        # if only want to train on a pre-trained model, set config_train_file to "none";
        # but if you accidentally leave config_train_file not blank, the default is not to overwrite, so it shouldn't matter!
        config_train_file="none"

        ### TESTING ###
        ### cosmic variance (quijote)
        # tag_params_test="_quijote_p0_n1000"
        # tag_biasparams_test="_b1000_p0_n1"
        # tag_mean="_mean"
        # tag_noise_test="_noise_unit_quijote_p0_n1000"
        # tag_Anoise_test="_Anmult_p0_n1"
        ### coverage
        tag_params_test="_test_p5_n1000"
        tag_biasparams_test="_biaszen_p4_n1000"
        tag_mean=""
        tag_noise_test="_noise_unit_test_p5_n1000"
        tag_Anoise_test="_Anmult_p5_n1000"
        ### no noise
        # tag_noise_test=""
        # tag_Anoise_test=""
        ### Muchisimocks test set 
        tag_data_test="muchisimocks${tag_stats}${tag_mask}${tag_params_test}${tag_biasparams_test}${tag_noise_test}${tag_Anoise_test}${tag_mean}"
        config_test_file="../configs/configs_test/config_TRAIN_${tag_data_train}_TEST_${tag_data_test}.yaml"

        ### OOD test set
        # data_mode="shame" 
        # tag_mock="_nbar0.00022"
        # tag_data_test="shame${tag_stats}${tag_mask}${tag_mock}"
        # config_test_file="../configs/configs_test/config_TRAIN_${tag_data_train}_TEST_${tag_data_test}.yaml"
        
        # no test
        #config_test_file="none"

        # Determine job name logic
        if [[ "$config_train_file" == "none" && "$config_test_file" == "none" ]]; then
            echo "Both config_train and config_test are none, skipping job."
            continue
        elif [[ "$config_train_file" == "none" ]]; then
            #config_base=$(basename "$config_test_file" .yaml)
            #job_name="run_inf_${config_base}"
            # bc filename too long!!!
            if [[ "$data_mode" == "shame" ]]; then
                job_name="run_inf_${tag_data_train}_TEST${tag_data_test}"
            else
                job_name="run_inf_${tag_data_train}_TEST${tag_params_test}"
            fi
        elif [[ "$config_test_file" == "none" ]]; then
            config_base=$(basename "$config_train_file" .yaml)
            job_name="run_inf_${config_base}"
        else
            config_base=$(basename "$config_test_file" .yaml)
            job_name="run_inf_traintest_${config_base}"
        fi

        # Get the absolute path to the scripts directory (where this script is located)
        scripts_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        # Ensure logs directory exists (SLURM needs it before the job runs)
        mkdir -p "${scripts_dir}/logs" || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

        sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${scripts_dir}/logs/${job_name}.out
##SBATCH --time=0:20:00 # quick tests (e.g. shame)
##SBATCH --time=1:00:00 #1h for testing on cosmic var test set or single OOD
#SBATCH --time=6:00:00 #2h for training -> 6 in case
##SBATCH --time=24:00:00 #24h for testing on coverage test set (24h is max time limit; some dont converge)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G #40gb for both training and testing (30 failed one time)

# Change to the scripts directory to ensure relative paths work
cd "${scripts_dir}" || { echo "ERROR: Failed to change to scripts directory ${scripts_dir}" >&2; exit 1; }

# Ensure logs directory exists
mkdir -p logs || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Running on node \${SLURMD_NODENAME}"
echo "Working directory: \$(pwd)"

echo "config_train_file: ${config_train_file}"
echo "config_test_file: ${config_test_file}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

# Build command arguments
args=()
$( [[ "$config_train_file" != "none" ]] && printf 'args+=("--config-train=%s")\n' "${config_train_file}" )
$( [[ "$config_test_file" != "none" ]] && printf 'args+=("--config-test=%s")\n' "${config_test_file}" )

if [ \${#args[@]} -eq 0 ]; then
    echo "ERROR: No arguments to pass to run_inference.py" >&2
    exit 1
fi

echo "python run_inference.py \${args[@]}"
echo "Number of arguments: \${#args[@]}"

python run_inference.py "\${args[@]}"
EOF

    done
done