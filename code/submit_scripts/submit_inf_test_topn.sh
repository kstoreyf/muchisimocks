#!/bin/bash
#
# Submit inference tests for top-N stall-passing sweep runs (nth_best_run).
# Sweep: noisy pk+bispec+pgm, kb0.25, tag_sweep=-rand30
#
# Usage:
#   bash submit_inf_test_topn.sh ood _nbar0.00011 [start end]
#   bash submit_inf_test_topn.sh ood _nbar0.00054
#   bash submit_inf_test_topn.sh fixed_cosmo_shame_mean
#   bash submit_inf_test_topn.sh fixed_cosmo_shame_sample
#
set -euo pipefail

code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
results_sbi_root="/scratch/kstoreyf/muchisimocks/results/results_sbi"

preset="${1:?usage: $0 <ood|fixed_cosmo_shame_mean|fixed_cosmo_shame_sample> [tag_mock] [start end]}"
shift || true

tag_mock=""
if [[ "${preset}" == "ood" ]]; then
    tag_mock="${1:?ood requires tag_mock e.g. _nbar0.00011}"
    shift || true
fi

tag_stats="_pk_bispec_pgm"
tag_masks="_kb0.25"
tag_params="_p5_n10000"
tag_biasparams="_biasnoisenest_p9_n320000"
tag_noise="_noise_unit${tag_params}"
tag_rp="_rp"
bx=32
n_train=10000
tag_sweep="-rand30"
overwrite_test=false

case "${preset}" in
    ood) batch_timeout_seconds=1800 ;;
    *)   batch_timeout_seconds=7200 ;;
esac

tag_data_train="_muchisimocks${tag_stats}${tag_masks}${tag_params}${tag_biasparams}${tag_noise}"
tag_inf_num="_bx${bx}_ntrain${n_train}"
sweep_name="${tag_data_train}${tag_rp}${tag_inf_num}_sweep${tag_sweep}"
sweep_dir="${results_sbi_root}/sbi${sweep_name}"

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

echo "preset=${preset} tag_mock=${tag_mock:-n/a}"
echo "sweep_dir=${sweep_dir}"
echo "best_runs (${n_runs}): ${run_ids[*]}"
echo "submitting nth_best_run=${start}..${end} (skip if samples already exist)"

mkdir -p "${code_dir}/logs" "${code_dir}/../configs/configs_test"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

cd "${code_dir}"

manifest=$(mktemp)
python - <<PY
from pathlib import Path
from generate_config_inference import (
    generate_test_config_ood,
    generate_test_config_from_preset,
    resolve_train_tag_bundle,
    resolve_test_scenario_tags,
    build_config_tag_test,
    build_tag_data,
    DEFAULT_CONFIGS_TEST_DIR,
)

preset = "${preset}"
tag_mock = "${tag_mock}"
start, end = ${start}, ${end}
sweep_name = "${sweep_name}"
sweep_dir = Path("${sweep_dir}")
run_ids = """${run_ids[*]}""".split()
train = resolve_train_tag_bundle("${tag_params}", "noisy")
stats = ["pk", "bispec", "pgm"]
masks = ["", "_kb0.25", ""]
cfg_dir = Path(DEFAULT_CONFIGS_TEST_DIR)
lines = []

for n in range(start, end + 1):
    run_id = run_ids[n]
    if preset == "ood":
        generate_test_config_ood(
            overwrite=True,
            statistics=stats,
            tags_mask=masks,
            n_train=${n_train},
            bx=${bx},
            tag_params=train["tag_params"],
            tag_biasparams=train["tag_biasparams"],
            tag_noise=train["tag_noise"],
            reparameterize=True,
            tag_sweep="${tag_sweep}",
            nth_best_run=n,
            tag_mock=tag_mock,
            data_mode_test="shame",
        )
        # Matches generate_test_config_ood: '_' + data_mode_test + tag_stats + masks + mock
        tag_data_test = "_shame_" + "_".join(stats) + "".join(masks) + tag_mock
        evaluate_mean = False
        data_mode_test = "shame"
        tag_inf_train = f"{sweep_name}/{run_id}"
    else:
        generate_test_config_from_preset(
            preset,
            tag_params="${tag_params}",
            noise_mode="noisy",
            overwrite=True,
            statistics=stats,
            tags_mask=masks,
            n_train=${n_train},
            bx=${bx},
            tag_sweep="${tag_sweep}",
            nth_best_run=n,
        )
        scenario = resolve_test_scenario_tags(preset, "noisy", "_shame_p0_n1000")
        tag_data_test = build_tag_data(
            "muchisimocks",
            stats,
            masks,
            "_shame_p0_n1000",
            scenario["tag_biasparams_test"],
            scenario["tag_noise_test"],
        )
        evaluate_mean = preset == "fixed_cosmo_shame_mean"
        data_mode_test = "muchisimocks"
        tag_inf_train = f"{sweep_name}/{run_id}"

    tag_test = build_config_tag_test(
        tag_inf_train,
        tag_data_test,
        "muchisimocks",
        data_mode_test,
        stats,
        masks,
        evaluate_mean=evaluate_mean,
        nth_best_run=n,
    )
    cfg_path = cfg_dir / f"config{tag_test}.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"ERROR: expected config missing: {cfg_path}")
    mean_suffix = "_mean" if evaluate_mean else ""
    samples_path = sweep_dir / run_id / f"samples_test{tag_data_test}{mean_suffix}_pred.npy"
    lines.append(f"{n}|{run_id}|{cfg_path}|{samples_path}")
    print(f"config nbest{n} {run_id}: {cfg_path.name}")
    print(f"  samples -> {samples_path.name}")

Path("${manifest}").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

overwrite_test_flag=""
if [[ "${overwrite_test}" == true ]]; then
    overwrite_test_flag="--overwrite-test"
fi

while IFS='|' read -r nth run_id config_test_file samples_path; do
    [[ -z "${nth}" ]] && continue
    if [[ -f "${samples_path}" && "${overwrite_test}" != true ]]; then
        echo "[nbest${nth}] ${run_id}: samples exist, skip — ${samples_path}"
        continue
    fi

    job_name="inf_${preset}_n${nth}_${run_id}"
    if [[ -n "${tag_mock}" ]]; then
        mock_short="${tag_mock#_nbar}"
        job_name="inf_ood_${mock_short}_n${nth}_${run_id}"
    fi
    if (( ${#job_name} > 64 )); then
        job_name="inf_n${nth}_${run_id}"
    fi

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
echo "preset=${preset} tag_mock=${tag_mock} nth_best_run=${nth} run_id=${run_id}"
echo "config_test_file=${config_test_file}"
echo "expected_samples=${samples_path}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

python run_inference.py --config-test="${config_test_file}" --batch-timeout-seconds=${batch_timeout_seconds} ${overwrite_test_flag}
EOF
)
    echo "[nbest${nth}] ${run_id}: ${submit_out}"
done < "${manifest}"

rm -f "${manifest}"
