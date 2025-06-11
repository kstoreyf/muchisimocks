#!/bin/bash

#n_train_arr=(500 1000 2000)
#n_train_arr=(500 1000 2000 4000 6000 8000 10000)
#tag_stats_arr=("_pk" "_bispec" "_pk_bispec") 

n_train_arr=(10000)
tag_stats_arr=("_pk_bispec")

for n_train in "${n_train_arr[@]}"; do
    for tag_stats in "${tag_stats_arr[@]}"; do
        #config_train_file="../configs/configs_train/config_muchisimocks${tag_stats}_p5_n10000_biaszen_p4_n10000_ntrain${n_train}.yaml"
        config_train_file="none"
        config_test_file="../configs/configs_test/config_TRAIN_muchisimocks${tag_stats}_p5_n10000_biaszen_p4_n100000_ntrain${n_train}_TEST_muchisimocks${tag_stats}_test_p5_n1000_biaszen_p4_n1000.yaml"
        # config_test_file="none"

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
#SBATCH --time=16:00:00 #2h for training, more for testing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G #30gb for training

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