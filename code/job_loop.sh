#!/bin/bash
# Submit choose_best_run.py once per tags_stat. Run from login: bash job_loop.sh


tag_stats_arr=("_pk" "_pk_pgm" "_pk_bispec" "_pk_bispec_pgm")

code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${code_dir}/logs"

for tag_stats in "${tag_stats_arr[@]}"; do
    ### fix wandb metrics
    if [[ "$tag_stats" == *bispec* ]]; then
        tag_masks="_kb0.25"
    else
        tag_masks=""
    fi
    sweep_name="_muchisimocks${tag_stats}${tag_masks}_p5_n10000_biasnest_p4_n320000_rp_bx32_ntrain10000_sweep-rand30"
    #sweep_name="_muchisimocks${tag_stats}${tag_masks}_p5_n10000_biasnoisenest_p9_n320000_noise_unit_p5_n10000_rp_bx32_ntrain10000_sweep-rand30"
    job_name="fix_wandb_metrics${sweep_name}"

    ### choose best run
    #noise_mode="noiseless"
    #noise_mode="noisy"
    #job_name="choose_best_run${tag_stats}_${noise_mode}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${code_dir}/logs/${job_name}.out
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=10G

cd "${code_dir}" || exit 1
echo "\$(date)  job=\${SLURM_JOB_ID}  tags_stat=${tag_stats}"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
#python choose_best_run.py --tags_stat "${tag_stats}" --noise-modes ${noise_mode}

python fix_wandb_metrics.py ${sweep_name} --write-config-pkl
EOF
done
