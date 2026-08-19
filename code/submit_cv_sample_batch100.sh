#!/bin/bash
# Submit fixed-cosmo CV sample inference with checkpoint_every=100, batch timeout 2h.
set -euo pipefail
code_dir=/home/kstoreyf/muchisimocks/code
cd "$code_dir"

configs=(
  "config_TRAIN_muchisimocks_pk_p5_n10000_biasnoisenest_p9_n320000_noise_unit_p5_n10000_rp_bx32_ntrain10000_best-rand30_TEST_shame_p0_n1000_biasshame_noisebest_p0_n1_noise_unit_shame_p0_n1000.yaml"
  "config_TRAIN_muchisimocks_pk_pgm_kpgm0.25_p5_n10000_biasnoisenest_p9_n320000_noise_unit_p5_n10000_rp_bx32_ntrain10000_best-rand30_TEST_shame_p0_n1000_biasshame_noisebest_p0_n1_noise_unit_shame_p0_n1000.yaml"
  "config_TRAIN_muchisimocks_pk_bispec_kb0.25_p5_n10000_biasnoisenest_p9_n320000_noise_unit_p5_n10000_rp_bx32_ntrain10000_best-rand30_TEST_shame_p0_n1000_biasshame_noisebest_p0_n1_noise_unit_shame_p0_n1000.yaml"
)

for cfg in "${configs[@]}"; do
  config_test_file="../configs/configs_test/${cfg}"
  short=$(echo "$cfg" | sed -E 's/config_TRAIN_muchisimocks_//;s/_p5_n10000.*//')
  job_name="inf_test_cv_sample_${short}"
  echo "Submitting $job_name"
  sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${code_dir}/logs/inf_test_%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G

cd "${code_dir}" || exit 1
mkdir -p logs
echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "tag_stats=${short}"
echo "Running on node \${SLURMD_NODENAME}"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
echo "python run_inference.py --config-test=${config_test_file} --checkpoint-every=100 --batch-timeout-seconds=7200"
python run_inference.py --config-test="${config_test_file}" --checkpoint-every=100 --batch-timeout-seconds=7200
EOF
done

squeue -u kstoreyf -o '%.18i %.28j %.2t %.10M %.10l'
