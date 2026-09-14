#!/bin/bash
# Train noiseless kp0.35 fiducial model, then run shame-mean test for fig 7.
set -euo pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_CFG="${CODE}/../configs/configs_train/config_muchisimocks_pk_bispec_pgm_kp0.35_kb0.25_kpgm0.25_p5_n10000_biasnest_p4_n320000_rp_bx32_ntrain10000_best-rand30.yaml"
TEST_CFG="${CODE}/../configs/configs_test/config_TRAIN_muchisimocks_pk_bispec_pgm_kp0.35_kb0.25_kpgm0.25_p5_n10000_biasnest_p4_n320000_rp_bx32_ntrain10000_best-rand30_TEST_shame_p0_n1000_biasshame_p0_n1_mean.yaml"
mkdir -p "${CODE}/logs"

TRAIN_OUT=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=inf_train_nl_kp035_fig7
#SBATCH --output=${CODE}/logs/inf_train_nl_kp035_fig7_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

cd "${CODE}" || exit 1
echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Running on node \${SLURMD_NODENAME}"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
echo "python run_inference.py --config-train=${TRAIN_CFG}"
python run_inference.py --config-train="${TRAIN_CFG}"
EOF
)
echo "TRAIN_JOB=${TRAIN_OUT}"

TEST_OUT=$(sbatch --parsable --dependency=afterok:${TRAIN_OUT} <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=inf_test_nl_kp035_fig7_mean
#SBATCH --output=${CODE}/logs/inf_test_nl_kp035_fig7_mean_%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

cd "${CODE}" || exit 1
echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Running on node \${SLURMD_NODENAME}"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
echo "python run_inference.py --config-test=${TEST_CFG}"
python run_inference.py --config-test="${TEST_CFG}"
EOF
)
echo "TEST_JOB=${TEST_OUT}"
squeue -u "${USER}" -j "${TRAIN_OUT},${TEST_OUT}"
