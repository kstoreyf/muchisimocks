#!/bin/bash
# Extra random trials for an existing v2 sweep, as a separate W&B sweep (same search space, own run_cap) writing into
# the same results dir. W&B refuses to raise run_cap of a started sweep, so this is how a sweep is extended.
# Arg: the extension train yaml: a copy of the sweep's train yaml with sweep_num_runs = n extra trials and no
# wandb_sweep_id (the first task creates the extension sweep and writes its id into that yaml).
#   sbatch --job-name=sweep_v2_<stats>_ext1 --array=0-<n-1> jobs/job_sweep_ext_v2.s <ext yaml>
# Same body as code/submit_scripts/submit_inf_train_sweep.sh (checkpoint, requeue, resume).
# Agents must run with MUCHISIMOCKS_SWEEP_PARALLEL=1 (set below): otherwise sbi_model deletes local trial dirs that
# are not in *its* W&B sweep, which would remove the other sweep's trials from the shared dir.
#SBATCH --qos=regular
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x_%A_%a.out

config_train_file=$1
code_dir=/home/kstoreyf/muchisimocks/code
cd "${code_dir}" || exit 1
echo "Current date and time: $(date)"
echo "Slurm job id is ${SLURM_JOB_ID}"
echo "Array task id is ${SLURM_ARRAY_TASK_ID} (restart ${SLURM_RESTART_COUNT:-0})"
echo "Running on node ${SLURMD_NODENAME}"
echo "config_train_file: ${config_train_file}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
source /home/kstoreyf/muchisimocks/code_auxiliary/env_v2.sh

export MUCHISIMOCKS_SWEEP_PARALLEL=1
export MUCHISIMOCKS_SWEEP_RUNS_PER_AGENT=1
export MUCHISIMOCKS_TRAIN_DEADLINE=$(( $(date +%s) + 24*3600 - 45*60 ))
sleep $(( SLURM_ARRAY_TASK_ID * 5 ))

python run_inference.py --config-train="${config_train_file}"
rc=$?
if [[ $rc -eq 3 ]]; then  # sbi_model.EXIT_TRAINING_INCOMPLETE: checkpointed, resume in a new job
    task="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "$(date): training incomplete; requeueing ${task}"
    scontrol requeue "${task}" && exit 0
    fn_resub="/scratch/kstoreyf/muchisimocks/logs_v2/resubmit_${SLURM_JOB_ID}.sh"
    scontrol write batch_script "${SLURM_JOB_ID}" "${fn_resub}" && sbatch --array="${SLURM_ARRAY_TASK_ID}" "${fn_resub}"
fi
exit $rc
