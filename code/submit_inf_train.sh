#!/bin/bash

#n_train_arr=(500 1000 2000 4000 6000 8000 10000)
n_train_arr=(10000)
#tag_stats_arr=("_pk") 
tag_stats_arr=("_pk" "_pk_pgm") 
#tag_stats_arr=("_pk_pgm") 
#tag_stats_arr=("_pk_bispec_pgm")
#tag_stats_arr=("_pk_bispec" "_pk_bispec_pgm")
#tag_stats_arr=("_pk" "_pk_pgm" "_pk_bispec" "_pk_bispec_pgm")

for n_train in "${n_train_arr[@]}"; do
    for tag_stats in "${tag_stats_arr[@]}"; do
        tag_params="_p5_n10000"
        bx=32
        #tag_biasparams="_biasnest_p4_n320000"  
        #tag_noise=""
        tag_biasparams="_biasnoisenest_p9_n320000"
        tag_noise="_noise_unit_p5_n10000"
        tag_masks=""
        #tag_masks="_kb0.25"
        tag_data_train="_muchisimocks${tag_stats}${tag_masks}${tag_params}${tag_biasparams}${tag_noise}"
        tag_rp="_rp"
        tag_inf_num="_bx${bx}_ntrain${n_train}"
        tag_sweep="_sweep-rand30"
        #tag_sweep=""
        tag_inf="${tag_data_train}${tag_rp}${tag_inf_num}${tag_sweep}"
        config_train_file="../configs/configs_train/config${tag_inf}.yaml"

        job_name="inf_train${tag_inf}"

        code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        mkdir -p "${code_dir}/logs" || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

        sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${code_dir}/logs/${job_name}.out
##SBATCH --time=0:20:00
##SBATCH --time=24:00:00 #0.5-2h for training -> 4 in case
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

cd "${code_dir}" || { echo "ERROR: Failed to change to code directory ${code_dir}" >&2; exit 1; }
mkdir -p logs || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Running on node \${SLURMD_NODENAME}"
echo "Working directory: \$(pwd)"
echo "config_train_file: ${config_train_file}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

echo "python run_inference.py --config-train=${config_train_file}"
python run_inference.py --config-train="${config_train_file}"
EOF

    done
done
