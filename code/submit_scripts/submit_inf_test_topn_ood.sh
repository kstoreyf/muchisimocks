#!/bin/bash
#
# Submit SHAMe OOD inference for top-N stall-passing sweep runs (nth_best_run).
# Matches notebooks/2026-06-04_sweep_top5_ensemble_shame.ipynb:
#   noisy pk+bispec+pgm, kb0.25, tag_mock=_nbar0.00022, tag_sweep=-rand30
#
# Usage (from repo root or code/):
#   bash submit_inf_test_topn_ood.sh            # all ranks in best_runs.txt missing samples
#   bash submit_inf_test_topn_ood.sh 5 9        # only nth_best_run 5..9
#
set -euo pipefail

code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
results_sbi_root="/scratch/kstoreyf/muchisimocks/results/results_sbi"

tag_stats="_pk_bispec_pgm"
tag_masks="_kb0.25"
tag_params="_p5_n10000"
tag_biasparams="_biasnoisenest_p9_n320000"
tag_noise="_noise_unit${tag_params}"
tag_rp="_rp"
bx=32
n_train=10000
tag_sweep="-rand30"
tag_mock="_nbar0.00022"
overwrite_test=false
batch_timeout_seconds=1800

tag_data_train="_muchisimocks${tag_stats}${tag_masks}${tag_params}${tag_biasparams}${tag_noise}"
tag_inf_num="_bx${bx}_ntrain${n_train}"
sweep_name="${tag_data_train}${tag_rp}${tag_inf_num}_sweep${tag_sweep}"
sweep_dir="${results_sbi_root}/sbi${sweep_name}"
tag_data_test="_shame${tag_stats}${tag_masks}${tag_mock}"
samples_name="samples_test${tag_data_test}_pred.npy"

if [[ ! -f "${sweep_dir}/best_runs.txt" ]]; then
    echo "ERROR: missing ${sweep_dir}/best_runs.txt" >&2
    exit 1
fi

mapfile -t run_ids < <(grep -v '^[[:space:]]*$' "${sweep_dir}/best_runs.txt")
n_runs=${#run_ids[@]}
if (( n_runs < 1 )); then
    echo "ERROR: best_runs.txt is empty" >&2
    exit 1
fi

if (( $# >= 2 )); then
    start=$1
    end=$2
elif (( $# == 1 )); then
    start=$1
    end=$((n_runs - 1))
else
    start=0
    end=$((n_runs - 1))
fi

if (( start < 0 || end >= n_runs || start > end )); then
    echo "ERROR: rank range ${start}..${end} invalid for ${n_runs} runs in best_runs.txt" >&2
    exit 1
fi

echo "sweep_dir=${sweep_dir}"
echo "best_runs (${n_runs}): ${run_ids[*]}"
echo "submitting nth_best_run=${start}..${end} (skip if samples already exist)"

mkdir -p "${code_dir}/logs" "${code_dir}/../configs/configs_test"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

cd "${code_dir}"
python - <<PY
from generate_config_inference import (
    generate_test_config_ood,
    resolve_train_tag_bundle,
)

train = resolve_train_tag_bundle("${tag_params}", "noisy")
for n in range(${start}, ${end} + 1):
    generate_test_config_ood(
        overwrite=True,
        statistics=["pk", "bispec", "pgm"],
        tags_mask=["", "_kb0.25", ""],
        n_train=${n_train},
        bx=${bx},
        tag_params=train["tag_params"],
        tag_biasparams=train["tag_biasparams"],
        tag_noise=train["tag_noise"],
        reparameterize=True,
        tag_sweep="${tag_sweep}",
        nth_best_run=n,
        tag_mock="${tag_mock}",
        data_mode_test="shame",
    )
    print(f"wrote config for nth_best_run={n} tag_mock=${tag_mock}")
PY

overwrite_test_flag=""
if [[ "${overwrite_test}" == true ]]; then
    overwrite_test_flag="--overwrite-test"
fi

for nth in $(seq "${start}" "${end}"); do
    run_id="${run_ids[$nth]}"
    samples_path="${sweep_dir}/${run_id}/${samples_name}"
    if [[ -f "${samples_path}" && "${overwrite_test}" != true ]]; then
        echo "[nbest${nth}] ${run_id}: samples exist, skip — ${samples_path}"
        continue
    fi

    tag_inf="${sweep_name}/${run_id}"
    tag_inf_file="${sweep_name}_nbest${nth}"
    tag_test="_TRAIN${tag_inf_file}_TEST${tag_data_test}"
    config_test_file="../configs/configs_test/config${tag_test}.yaml"
    if [[ ! -f "${config_test_file}" ]]; then
        echo "ERROR: missing ${config_test_file}" >&2
        exit 1
    fi

    job_name="inf_test_ood_noisy_nbest${nth}_${run_id}"
    submit_out=$(sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${code_dir}/logs/inf_test_%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G

cd "${code_dir}" || exit 1
echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "nth_best_run=${nth} run_id=${run_id}"
echo "config_test_file=${config_test_file}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

python run_inference.py --config-test="${config_test_file}" --batch-timeout-seconds=${batch_timeout_seconds} ${overwrite_test_flag}
EOF
)
    echo "[nbest${nth}] ${run_id}: ${submit_out}"
done
