#!/bin/bash

#n_train_arr=(500 1000 2000 4000 6000 8000 10000)
n_train_arr=(10000)
#tag_stats_arr=("_pk" "_bispec" "_pk_bispec") 
#tag_stats_arr=("_pk") 
#tag_stats_arr=("_bispec" "_pk_bispec") 
#tag_stats_arr=("_pk_bispec")
tag_stats_arr=("_pk_pgm")
#tag_stats_arr=("_pk_bispec_pgm")
#tag_stats_arr=("_pgm")
#tag_stats_arr=("_pgm" "_pk_pgm" "_pk_bispec_pgm") 
#tag_stats_arr=("_pk" "_pgm" "_bispec" "_pk_pgm" "_pk_bispec" "_pk_bispec_pgm")

for n_train in "${n_train_arr[@]}"; do
    for tag_stats in "${tag_stats_arr[@]}"; do
        ### TRAINING (for referencing the trained model) ###
        tag_params="_p5_n10000"
        tag_biasparams="_biasnest_p4_n320000"  
        tag_noise=""
        tag_mask=""
        #tag_mask="_kb0.25"
        tag_data_train="_muchisimocks${tag_stats}${tag_mask}${tag_params}${tag_biasparams}${tag_noise}"
        tag_rp="_rp"
        #tag_rp=""
        bx=4
        tag_inf_train="_bx${bx}_ntrain${n_train}"
        tag_inf="${tag_data_train}${tag_rp}${tag_inf_train}"

        ### TESTING ###
        ### cosmic variance (quijote)
        tag_params_test="_shame_p0_n1000"
        tag_biasparams_test="_biasshame_p0_n1"
        tag_mean="_mean"
        #tag_mean=""
        tag_noise_test=""
        ### coverage
        # tag_params_test="_test_p5_n1000"
        # #tag_biasparams_test="_b1000_p0_n1"
        # tag_biasparams_test="_biaszen_p4_n1000"
        # tag_mean=""
        # tag_noise_test="_noise_unit_test_p5_n1000"
        # tag_Anoise_test="_Anmult_p5_n1000"
        ## no noise
        # tag_noise_test=""
        # tag_Anoise_test=""
        ### Muchisimocks test set 
        tag_data_test="_muchisimocks${tag_stats}${tag_mask}${tag_params_test}${tag_biasparams_test}${tag_noise_test}${tag_mean}"

        ### OOD test set
        #data_mode="shame" 
        #tag_mock="_nbar0.00022"
        #tag_data_test="_shame${tag_stats}${tag_mask}${tag_mock}"

        config_test_file="../configs/configs_test/config_TRAIN${tag_inf}_TEST${tag_data_test}.yaml"

        job_name="inf_test_TRAIN${tag_inf}_TEST${tag_data_test}"

        code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        mkdir -p "${code_dir}/logs" || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

        sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${code_dir}/logs/${job_name}.out
#SBATCH --time=0:20:00 # quick tests (e.g. shame)
##SBATCH --time=1:30:00 #1h for testing on cosmic var test set or single OOD
##SBATCH --time=24:00:00 #24h for testing on coverage test set (24h is max time limit; some dont converge)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

cd "${code_dir}" || { echo "ERROR: Failed to change to code directory ${code_dir}" >&2; exit 1; }
mkdir -p logs || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Running on node \${SLURMD_NODENAME}"
echo "Working directory: \$(pwd)"
echo "config_test_file: ${config_test_file}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

echo "python run_inference.py --config-test=${config_test_file}"
python run_inference.py --config-test="${config_test_file}"
EOF

    done
done
