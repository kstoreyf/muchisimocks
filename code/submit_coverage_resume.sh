#!/bin/bash
# Resume incomplete coverage inference (NaN holes in in-progress 1000-obs files).
#
# Batch size 1 so a stall only burns that one obs (previous runs used 5, which
# is why holes come in groups of ~5). Timeout 1 h/obs — healthy single-obs
# sampling is ~1–2 min; 1 h is enough for slow cases without sitting on a
# freeze for the old 2 h / batch-of-5 budget. Walltime 24 h (regular QOS max);
# worst leftover is 20 pending obs.
set -euo pipefail
code_dir=/home/kstoreyf/muchisimocks/code
cd "$code_dir"
mkdir -p logs

checkpoint_every=1
batch_timeout_seconds=3600
walltime=24:00:00

mapfile -t configs < <(python3 - << 'PY'
import re
from pathlib import Path
import numpy as np

base = Path("/scratch/kstoreyf/muchisimocks/results/results_sbi")
cfg_dir = Path("/home/kstoreyf/muchisimocks/configs/configs_test")
rows = []
for fn in sorted(base.glob("sbi*_best-rand30/samples_test*coverage_p5_n1000*_pred*.npy")):
    if "_nbest" in fn.parent.name:
        continue
    model = fn.parent.name
    if "biasnoisenest_p9" not in model:
        continue
    kind = "inprogress" if "inprogress" in fn.name else "done"
    arr = np.load(fn, mmap_mode="r")
    if arr.ndim == 2:
        n_usable = int(np.any(np.isfinite(arr)))
    else:
        n_usable = int(np.any(np.isfinite(arr[0]), axis=1).sum())
    if n_usable >= 1000 and kind == "done":
        continue
    mid = model[len("sbi_"):]
    cfg_name = (
        f"config_TRAIN_{mid}"
        f"_TEST_coverage_p5_n1000_biasnoisecoverage_p9_n1000_noise_unit_coverage_p5_n1000.yaml"
    )
    cfg = cfg_dir / cfg_name
    if not cfg.is_file():
        raise SystemExit(f"missing config: {cfg}")
    pending = 1000 - n_usable
    rows.append((pending, cfg_name, model, n_usable))

rows.sort()
for pending, cfg_name, model, n_usable in rows:
    print(f"{cfg_name}\t{n_usable}\t{pending}")
PY
)

echo "Submitting ${#configs[@]} coverage resume jobs"
echo "  checkpoint_every=${checkpoint_every}  batch_timeout_seconds=${batch_timeout_seconds}  time=${walltime}"
echo

for line in "${configs[@]}"; do
  cfg="${line%%$'\t'*}"
  rest="${line#*$'\t'}"
  n_usable="${rest%%$'\t'*}"
  pending="${rest#*$'\t'}"
  config_test_file="../configs/configs_test/${cfg}"
  short=$(echo "$cfg" | sed -E 's/config_TRAIN_muchisimocks_//;s/_p5_n10000.*_rp//;s/_best-rand30_TEST.*//')
  job_name="cov_resume_${short}"
  if (( ${#job_name} > 60 )); then
    job_name="cov_resume_$(printf '%s' "$short" | md5sum | awk '{print substr($1,1,12)}')"
  fi
  echo "Submitting ${job_name}  usable=${n_usable}/1000  pending=${pending}"
  sbatch <<EOF
#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=${job_name}
#SBATCH --output=${code_dir}/logs/inf_test_%j.out
#SBATCH --time=${walltime}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G

cd "${code_dir}" || exit 1
mkdir -p logs
echo "Current date and time: \$(date)"
echo "Slurm job id is \${SLURM_JOB_ID}"
echo "Running on node \${SLURMD_NODENAME}"
echo "coverage resume: usable=${n_usable}/1000 pending=${pending}"
echo "checkpoint_every=${checkpoint_every} batch_timeout_seconds=${batch_timeout_seconds}"
echo "config_test_file: ${config_test_file}"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
echo "python run_inference.py --config-test=${config_test_file} --checkpoint-every=${checkpoint_every} --batch-timeout-seconds=${batch_timeout_seconds}"
python run_inference.py --config-test="${config_test_file}" --checkpoint-every=${checkpoint_every} --batch-timeout-seconds=${batch_timeout_seconds}
EOF
done

echo
squeue -u kstoreyf -o '%.18i %.40j %.2t %.10M %.10l' | head -50
echo "... ($(squeue -u kstoreyf -h | wc -l) jobs in queue)"
