#!/bin/bash

n_train_arr=(10000)
bx_arr=(32)
#n_train_arr=(500 1000 2000 4000 6000 8000 10000)
#bx_arr=(1 2 4 8 16 32)
#n_train_arr=(80000 10000)
#n_train_arr=(500 1000 2000 4000 6000)
#bx_arr=(8 16 32)
#bx_arr=(16 32)
#bx_arr=(1 2 4)
#bx_arr=(8 16 32)
tag_stats_arr=("_pk_bispec_pgm")
#tag_stats_arr=("_pk_bispec")
#tag_stats_arr=("_pk_pgm") 
#tag_stats_arr=("_pk" "_pk_pgm") 
#tag_stats_arr=("_pk" "_pk_pgm" "_pk_bispec") 
#tag_masks_arr=("" "_kpgm0.25" "_kb0.25")
#tag_stats_arr=("_pk_bispec_pgm")
tag_masks_arr=("_kb0.25_kpgm0.25")
#tag_masks_arr=("")
# _pk_bispec_pgm: same grid as generate_config_inference.py main() (tags_mask = ["", kb, kpgm] → tag_masks = kb+kpgm).
# tag_mask_bispec_arr=("_kb0.1" "_kb0.15" "_kb0.2" "_kb0.25" "_kb0.3" "_kb0.35" "")
# tag_mask_pgm_arr=("_kpgm0.1" "_kpgm0.15" "_kpgm0.2" "_kpgm0.25" "_kpgm0.3" "_kpgm0.35" "")
# tag_masks_arr=()

# for kb in "${tag_mask_bispec_arr[@]}"; do
#     for kpgm in "${tag_mask_pgm_arr[@]}"; do
#         tag_masks_arr+=("${kb}${kpgm}")
#     done
# done
#tag_stats_arr=("_pk_pgm" "_pk_bispec_pgm")
#tag_stats_arr=("_pk_pgm")

# Parallel W&B sweep agents (ignored unless tag_sweep is _sweep-*). Each array
# task trains one trial; they share one sweep via wandb_sweep_id in the yaml.
# For a 30-trial sweep, set n_sweep_agents=30 and tag_sweep="_sweep-rand30".
n_sweep_agents=1

for n_train in "${n_train_arr[@]}"; do
    for bx in "${bx_arr[@]}"; do
        for tag_stats in "${tag_stats_arr[@]}"; do
            for tag_masks in "${tag_masks_arr[@]}"; do
                tag_params="_p5_n10000"
                tag_biasparams="_biasnest_p4_n320000"  
                tag_noise=""
                #tag_biasparams="_biasnoisenest_p9_n320000"
                #tag_biasparams="_biasnoisem2nest_p7_n320000"
                #tag_noise="_noise_unit_p5_n10000"
                #if [[ "$tag_stats" != *bispec* ]]; then # fiducial masks
                # if [[ "$tag_stats" != *pgm* ]]; then # fiducial masks
                #     tag_masks=""
                # fi
                tag_data_train="_muchisimocks${tag_stats}${tag_masks}${tag_params}${tag_biasparams}${tag_noise}"
                tag_rp="_rp"
                tag_inf_num="_bx${bx}_ntrain${n_train}"
                #tag_sweep="_sweep-rand30"
                tag_sweep="_best-rand30"
                #tag_sweep=""
                tag_inf="${tag_data_train}${tag_rp}${tag_inf_num}${tag_sweep}"
                config_train_file="../configs/configs_train/config${tag_inf}.yaml"

                job_name="inf_train${tag_inf}"

                submit_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
                code_dir=$(cd "${submit_dir}/.." && pwd)
                mkdir -p "${submit_dir}/logs" || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }
                config_train_file="${code_dir}/../configs/configs_train/config${tag_inf}.yaml"
                if [[ ! -f "${config_train_file}" ]]; then
                    echo "ERROR: missing train config ${config_train_file}" >&2
                    exit 1
                fi

                array_line=""
                log_file="${submit_dir}/logs/${job_name}.out"
                if [[ "${n_sweep_agents}" -gt 1 ]]; then
                    array_line="#SBATCH --array=0-$((n_sweep_agents - 1))"
                    log_file="${submit_dir}/logs/${job_name}_%A_%a.out"
                fi

                sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_file}
${array_line}
##SBATCH --time=0:20:00
#SBATCH --time=24:00:00 
##SBATCH --time=48:00:00
##SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

cd "${code_dir}" || { echo "ERROR: Failed to change to code directory ${code_dir}" >&2; exit 1; }
mkdir -p "${submit_dir}/logs" || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Array task id is \${SLURM_ARRAY_TASK_ID:-none}"
echo "Running on node \${SLURMD_NODENAME}"
echo "Working directory: \$(pwd)"
echo "config_train_file: ${config_train_file}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

if [[ "${n_sweep_agents}" -gt 1 ]]; then
    export MUCHISIMOCKS_SWEEP_PARALLEL=1
    export MUCHISIMOCKS_SWEEP_RUNS_PER_AGENT=1
    sleep \$(( \${SLURM_ARRAY_TASK_ID:-0} * 2 ))
fi

echo "python run_inference.py --config-train=${config_train_file}"
python run_inference.py --config-train="${config_train_file}"
EOF

            done
        done
    done
done
