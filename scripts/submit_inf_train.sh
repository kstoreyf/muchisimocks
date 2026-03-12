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
        tag_params="_p5_n10000"
        tag_biasparams="_biasnest_p4_n320000"  
        bx=1
        tag_noise=""
        tag_Anoise=""   
        tag_mask=""
        #tag_mask="_kb0.25"
        tag_data_train="_muchisimocks${tag_stats}${tag_mask}${tag_params}${tag_biasparams}${tag_noise}${tag_Anoise}"
        #tag_rp=""
        tag_rp="_rp"
        tag_inf_train="_bx${bx}_ntrain${n_train}"
        tag_inf="${tag_data_train}${tag_rp}${tag_inf_train}"
        config_train_file="../configs/configs_train/config${tag_inf}.yaml"

        job_name="inf_train${tag_inf}"

        scripts_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
        mkdir -p "${scripts_dir}/logs" || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

        sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${scripts_dir}/logs/${job_name}.out
##SBATCH --time=0:20:00
#SBATCH --time=4:00:00 #0.5-2h for training -> 4 in case
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

cd "${scripts_dir}" || { echo "ERROR: Failed to change to scripts directory ${scripts_dir}" >&2; exit 1; }
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
