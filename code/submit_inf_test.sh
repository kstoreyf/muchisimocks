#!/bin/bash
#
# Submit inference test jobs. Test scenarios and tag rules mirror
# ``code/generate_config_inference.py``: ``PARAM_SETS_TEST``, ``_TEST_SCENARIO_TAGS``,
# and ``resolve_train_tag_bundle`` / ``resolve_test_scenario_tags``.
#
# Pick one or more presets (keys of PARAM_SETS_TEST):
#   coverage                 — PARAM_SETS_TEST["coverage"]
#   fixed_cosmo_shame_mean   — shame + evaluate_mean (tag_mean=_mean in filename)
#   fixed_cosmo_shame_sample — shame, per-realization
#   ood                      — shame OOD (tag_mock from preset); train noise_mode still
#                              selects which checkpoint; test tag string is unchanged
#
# Pick noise_mode (matches training + in-dist test bias/noise tags):
#   noiseless | noisy
#

# --- test matrix (edit these) ---
#test_preset_arr=(coverage)
test_preset_arr=(ood)
#test_preset_arr=(fixed_cosmo_shame_mean)
#test_preset_arr=(fixed_cosmo_shame_mean)
#test_preset_arr=(coverage fixed_cosmo_shame_mean ood)
#test_preset_arr=(coverage ood)
#test_preset_arr=(coverage fixed_cosmo_shame_mean fixed_cosmo_shame_sample ood)

noise_mode_arr=(noisy)
#noise_mode_arr=(noiseless)
# noise_mode_arr=(noiseless noisy)

# --- training / stats grid (same as before) ---
n_train_arr=(10000)
bx_arr=(32)
#n_train_arr=(500 1000 2000 4000 6000 8000 10000)
#bx_arr=(1 2 4 8 16 32)

#tag_stats_arr=("_pk")
#tag_stats_arr=("_pk_pgm")
#tag_stats_arr=("_pk_bispec" "_pk_bispec_pgm")
#tag_stats_arr=("_pk_bispec_pgm")
#tag_stats_arr=("_pk_bispec")
#tag_stats_arr=("_pk" "_pk_pgm" "_pk_bispec_pgm")
#tag_stats_arr=("_pk" "_pk_pgm")
#tag_stats_arr=("_pk_pgm" "_pk_bispec")
#tag_stats_arr=("_pk_bispec" "_pk_bispec_pgm")
tag_stats_arr=("_pk" "_pk_pgm" "_pk_bispec" "_pk_bispec_pgm")
tag_masks_arr=("_kb0.15" "_kb0.2" "_kb0.3" "_kb0.35" "_kb0.4")

# Train cosmo LH tag (must match generated configs)
tag_params_train="_p5_n10000"

tag_rp="_rp"
#tag_rp=""
tag_sweep="_best-rand30"
#tag_sweep=""

# ---------------------------------------------------------------------------
# Train-side tags: same strings as ``resolve_train_tag_bundle`` / NOISE_MODE_TRAIN_BIAS
# in generate_config_inference.py
# ---------------------------------------------------------------------------
set_train_tags_from_noise_mode() {
    local noise_mode="$1"
    tag_params="${tag_params_train}"
    case "${noise_mode}" in
        noiseless)
            tag_biasparams="_biasnest_p4_n320000"
            tag_noise=""
            ;;
        noisy)
            tag_biasparams="_biasnoisenest_p9_n320000"
            tag_noise="_noise_unit${tag_params}"
            ;;
        *)
            echo "ERROR: unknown noise_mode=${noise_mode}; expected noiseless or noisy" >&2
            return 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Test-side tags: same as PARAM_SETS_TEST + _TEST_SCENARIO_TAGS +
# resolve_test_scenario_tags in generate_config_inference.py.
# Sets: tag_params_test, tag_biasparams_test, tag_noise_test, tag_mean, tag_data_test
# tag_data_test excludes tag_mean (tag_mean is only appended to config filename, like tag_test).
# ---------------------------------------------------------------------------
set_test_tags_from_preset() {
    local preset="$1"
    local noise_mode="$2"

    tag_params_test=""
    tag_biasparams_test=""
    tag_noise_test=""
    tag_mean=""

    case "${preset}" in
        coverage)
            # PARAM_SETS_TEST["coverage"], _TEST_SCENARIO_TAGS["coverage"]
            tag_params_test="_coverage_p5_n1000"
            if [[ "${noise_mode}" == "noiseless" ]]; then
                tag_biasparams_test="_biascoverage_p4_n1000"
                tag_noise_test=""
            else
                tag_biasparams_test="_biasnoisecoverage_p9_n1000"
                tag_noise_test="_noise_unit${tag_params_test}"
            fi
            tag_data_test="_muchisimocks${tag_stats}${tag_masks}${tag_params_test}${tag_biasparams_test}${tag_noise_test}"
            ;;
        fixed_cosmo_shame_mean)
            # evaluate_mean=True -> tag_mean=_mean on filename only
            tag_params_test="_shame_p0_n1000"
            tag_mean="_mean"
            tag_biasparams_test="_biasshame_p0_n1"
            if [[ "${noise_mode}" == "noiseless" ]]; then
                tag_noise_test=""
            else
                tag_noise_test="_noise_unit${tag_params_test}"
            fi
            tag_data_test="_muchisimocks${tag_stats}${tag_masks}${tag_params_test}${tag_biasparams_test}${tag_noise_test}"
            ;;
        fixed_cosmo_shame_sample)
            tag_params_test="_shame_p0_n1000"
            tag_biasparams_test="_biasshame_p0_n1"
            if [[ "${noise_mode}" == "noiseless" ]]; then
                tag_noise_test=""
            else
                tag_noise_test="_noise_unit${tag_params_test}"
            fi
            tag_data_test="_muchisimocks${tag_stats}${tag_masks}${tag_params_test}${tag_biasparams_test}${tag_noise_test}"
            ;;
        ood)
            # PARAM_SETS_TEST["ood"]; generate_test_config_ood — no test noise/bias LH tags
            #local tag_mock="_nbar0.00022"
            local tag_mock="_nbar0.00054"
            tag_data_test="_shame${tag_stats}${tag_masks}${tag_mock}"
            ;;
        *)
            echo "ERROR: unknown test preset '${preset}'. Expected one of: coverage fixed_cosmo_shame_mean fixed_cosmo_shame_sample ood" >&2
            return 1
            ;;
    esac
}

for n_train in "${n_train_arr[@]}"; do
    for bx in "${bx_arr[@]}"; do
        for tag_stats in "${tag_stats_arr[@]}"; do
            for tag_masks in "${tag_masks_arr[@]}"; do
                for test_preset in "${test_preset_arr[@]}"; do
                    for noise_mode in "${noise_mode_arr[@]}"; do

                        if ! set_train_tags_from_noise_mode "${noise_mode}"; then
                            exit 1
                        fi
                        # only set given mask name when bispec is present
                        if [[ "$tag_stats" != *bispec* ]]; then 
                            tag_masks=""
                        fi
                        if ! set_test_tags_from_preset "${test_preset}" "${noise_mode}"; then
                            exit 1
                        fi

                        tag_data_train="_muchisimocks${tag_stats}${tag_masks}${tag_params}${tag_biasparams}${tag_noise}"
                        tag_inf_num="_bx${bx}_ntrain${n_train}"
                        tag_inf="${tag_data_train}${tag_rp}${tag_inf_num}${tag_sweep}"

                        # config stem matches generate_test_config / generate_test_config_ood: tag_test = _TRAIN..._TEST${tag_data_test}${tag_mean}
                        config_test_file="../configs/configs_test/config_TRAIN${tag_inf}_TEST${tag_data_test}${tag_mean}.yaml"

                        job_name="inf_test_${test_preset}_${noise_mode}_TRAIN${tag_inf}_TEST${tag_data_test}${tag_mean}"
                        # Slurm --output basename must fit NAME_MAX (typically 255). Drop redundant
                        # coverage_noisy prefix when the log filename would be too long.
                        if (( ${#job_name} + 4 > 255 )); then
                            job_name="inf_test_TRAIN${tag_inf}_TEST${tag_data_test}${tag_mean}"
                        fi

                        code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
                        mkdir -p "${code_dir}/logs" || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

                        sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${code_dir}/logs/${job_name}.out
##SBATCH --time=0:30:00
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G

cd "${code_dir}" || { echo "ERROR: Failed to change to code directory ${code_dir}" >&2; exit 1; }
mkdir -p logs || { echo "ERROR: Failed to create logs directory" >&2; exit 1; }

echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Running on node \${SLURMD_NODENAME}"
echo "Working directory: \$(pwd)"
echo "test_preset=${test_preset} noise_mode=${noise_mode}"
echo "config_test_file: ${config_test_file}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

echo "python run_inference.py --config-test=${config_test_file}"
python run_inference.py --config-test="${config_test_file}"
EOF
                    done
                done
            done
        done
    done
done
