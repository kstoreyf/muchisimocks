#!/bin/bash
#
# Retrain top-5 val-loss noisy pk+pgm sweep hparams at fiducial kpgm0.25,
# then run SHAMe OOD + CV-mean tests.
#
# Usage:
#   bash submit_inf_train_test_topn_pkpgm.sh
#
set -euo pipefail

code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${code_dir}"
mkdir -p logs "${code_dir}/../configs/configs_train" "${code_dir}/../configs/configs_test"

N_BEST=5

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

manifest=$(mktemp)
python - <<PY
import shutil
from pathlib import Path

import paths
from generate_config_inference import (
    DEFAULT_CONFIGS_TEST_DIR,
    DEFAULT_CONFIGS_TRAIN_DIR,
    build_config_tag_test,
    build_tag_data,
    generate_test_config_from_preset,
    generate_train_config,
    resolve_test_scenario_tags,
    resolve_train_tag_bundle,
)

n_best = ${N_BEST}
stats = ["pk", "pgm"]
masks = ["", "_kpgm0.25"]
tag_sweep = "-rand30"
tag_params = "_p5_n10000"
noise_mode = "noisy"
n_train = 10000
bx = 32
tag_mock = "_nbar0.00022"
data_mode = "muchisimocks"

train = resolve_train_tag_bundle(tag_params, noise_mode)
tag_data = build_tag_data(
    data_mode, stats, masks, train["tag_params"], train["tag_biasparams"], train["tag_noise"]
)
base_inf = tag_data + f"_rp_bx{bx}_ntrain{n_train}"
tag_data_test_ood = "_shame_" + "_".join(stats) + "".join(masks) + tag_mock
cv = resolve_test_scenario_tags("fixed_cosmo_shame_mean", noise_mode, "_shame_p0_n1000")
tag_data_test_cv = build_tag_data(
    data_mode,
    stats,
    masks,
    "_shame_p0_n1000",
    cv["tag_biasparams_test"],
    cv["tag_noise_test"],
)

results_sbi = Path(paths.DIR_RESULTS) / "results_sbi"
legacy_dir = results_sbi / f"sbi{base_inf}_best{tag_sweep}"
lines = []
for n in range(n_best):
    generate_train_config(
        overwrite=True,
        statistics=stats,
        tags_mask=masks,
        n_train=n_train,
        bx=bx,
        data_mode=data_mode,
        run_mode="best",
        tag_sweep=tag_sweep,
        nth_best_run=n,
        **train,
    )
    generate_test_config_from_preset(
        "ood",
        tag_params=tag_params,
        noise_mode=noise_mode,
        overwrite=True,
        statistics=stats,
        n_train=n_train,
        bx=bx,
        tags_mask=masks,
        tag_sweep=tag_sweep,
        nth_best_run=n,
        use_retrained_nbest=True,
        tag_mock=tag_mock,
    )
    generate_test_config_from_preset(
        "fixed_cosmo_shame_mean",
        tag_params=tag_params,
        noise_mode=noise_mode,
        overwrite=True,
        statistics=stats,
        n_train=n_train,
        bx=bx,
        tags_mask=masks,
        tag_sweep=tag_sweep,
        nth_best_run=n,
        use_retrained_nbest=True,
    )
    tag_inf = f"{base_inf}_best{tag_sweep}_nbest{n}"
    tag_test_ood = build_config_tag_test(
        tag_inf, tag_data_test_ood, data_mode, "shame", stats, masks,
        evaluate_mean=False, nth_best_run=n,
    )
    tag_test_cv = build_config_tag_test(
        tag_inf, tag_data_test_cv, data_mode, data_mode, stats, masks,
        evaluate_mean=True, nth_best_run=n,
    )
    dir_sbi = results_sbi / f"sbi{tag_inf}"
    fn_train = Path(DEFAULT_CONFIGS_TRAIN_DIR) / f"config{tag_inf}.yaml"
    fn_ood = Path(DEFAULT_CONFIGS_TEST_DIR) / f"config{tag_test_ood}.yaml"
    fn_cv = Path(DEFAULT_CONFIGS_TEST_DIR) / f"config{tag_test_cv}.yaml"
    fn_post = dir_sbi / "posterior.p"
    fn_samples_ood = dir_sbi / f"samples_test{tag_data_test_ood}_pred.npy"
    fn_samples_cv = dir_sbi / f"samples_test{tag_data_test_cv}_mean_pred.npy"

    # nbest0 == already-trained _best-rand30 (same hparams + kpgm0.25 data).
    if n == 0 and legacy_dir.is_dir():
        dir_sbi.mkdir(parents=True, exist_ok=True)
        for fn in ["posterior.p", "inference.p", "config.pkl", "param_names.txt"]:
            src, dst = legacy_dir / fn, dir_sbi / fn
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)
        for src in legacy_dir.glob("scaler_y_*.p"):
            dst = dir_sbi / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
        for src_name, dst in (
            (fn_samples_ood.name, fn_samples_ood),
            (fn_samples_cv.name, fn_samples_cv),
        ):
            src = legacy_dir / src_name
            if src.is_file() and not dst.is_file():
                shutil.copy2(src, dst)

    for p, kind in ((fn_train, "train"), (fn_ood, "ood"), (fn_cv, "cv")):
        if not p.is_file():
            raise SystemExit(f"missing {kind} config: {p}")
    lines.append(
        "|".join(
            [
                str(n),
                tag_inf,
                str(fn_train),
                str(fn_ood),
                str(fn_cv),
                str(fn_post),
                str(fn_samples_ood),
                str(fn_samples_cv),
            ]
        )
    )
    print(f"nbest{n} tag_inf={tag_inf}")
    print(f"  train {fn_train.name}")
    print(f"  ood   {fn_ood.name}")
    print(f"  cv    {fn_cv.name}")

Path("${manifest}").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "manifest=${manifest}"

submit_sbatch() {
    local out
    out=$(sbatch "$@")
    echo "${out}" >&2
    echo "${out}" | awk '{print $NF}'
}

while IFS='|' read -r n tag_inf fn_train fn_ood fn_cv fn_post fn_samples_ood fn_samples_cv; do
    [[ -z "${n}" ]] && continue
    train_jid=""
    if [[ -f "${fn_post}" ]]; then
        echo "[nbest${n}] posterior exists, skip train — ${fn_post}"
    else
        train_jid=$(submit_sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=pkpgm_tr_nb${n}
#SBATCH --output=${code_dir}/logs/pkpgm_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

cd "${code_dir}" || exit 1
echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "nth_best_run=${n}"
echo "config_train_file=${fn_train}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

python run_inference.py --config-train="${fn_train}"
EOF
)
        echo "[nbest${n}] train job ${train_jid}"
    fi

    dep=()
    if [[ -n "${train_jid}" ]]; then
        dep=(--dependency=afterok:${train_jid})
    fi

    if [[ -f "${fn_samples_ood}" ]]; then
        echo "[nbest${n}] OOD samples exist, skip test — ${fn_samples_ood}"
    else
        ood_jid=$(submit_sbatch "${dep[@]}" <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=pkpgm_te_ood_nb${n}
#SBATCH --output=${code_dir}/logs/pkpgm_%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G

cd "${code_dir}" || exit 1
echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "nth_best_run=${n} preset=ood"
echo "config_test_file=${fn_ood}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

python run_inference.py --config-test="${fn_ood}" --batch-timeout-seconds=1800
EOF
)
        echo "[nbest${n}] ood test job ${ood_jid}"
    fi

    if [[ -f "${fn_samples_cv}" ]]; then
        echo "[nbest${n}] CV-mean samples exist, skip test — ${fn_samples_cv}"
    else
        cv_jid=$(submit_sbatch "${dep[@]}" <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=pkpgm_te_cv_nb${n}
#SBATCH --output=${code_dir}/logs/pkpgm_%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G

cd "${code_dir}" || exit 1
echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "nth_best_run=${n} preset=fixed_cosmo_shame_mean"
echo "config_test_file=${fn_cv}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

python run_inference.py --config-test="${fn_cv}" --batch-timeout-seconds=7200
EOF
)
        echo "[nbest${n}] cv test job ${cv_jid}"
    fi
done < "${manifest}"

rm -f "${manifest}"
echo "done submitting pk+pgm top-${N_BEST} train/test jobs"
