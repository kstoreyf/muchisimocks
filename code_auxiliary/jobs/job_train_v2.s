#!/bin/bash
# v2 SBI training + SHAMe OOD tests, one array task per model.
# List file: one line per model, "<train config> <test config> [<test config> ...]" (paths relative to code/);
# train config "none" = test only (model already trained).
#   sbatch --job-name=<purpose> --array=0-5 jobs/job_train_v2.s jobs/lists/noise_ab_v2.txt      (submit from code_auxiliary/)
# Training checkpoints and stops 45 min before the wall time (MUCHISIMOCKS_TRAIN_DEADLINE); the task then
# requeues itself and resumes (as code/submit_scripts/submit_inf_train_sweep.sh). Finished trainings/tests are skipped on rerun.
#SBATCH --qos=regular
#SBATCH --job-name=train_v2
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
##SBATCH --gres=gpu:1  # not needed for training
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1   # training is single-threaded (bench_train_threads.py: more threads are slower)
#SBATCH --mem=30G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%A_%a.out

list=$(realpath $1)
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${list})
read -r config_train configs_test <<< "${line}"
echo "$(date) | job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} (restart ${SLURM_RESTART_COUNT:-0}) node ${SLURMD_NODENAME}"
echo "train: ${config_train}"

. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
cd /home/kstoreyf/muchisimocks/code
source /home/kstoreyf/muchisimocks/code_auxiliary/env_v2.sh

if [ "${config_train}" != "none" ]; then
    # stop training (with a checkpoint) 45 min before this job's wall time
    end=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j ${SLURM_JOB_ID} -o %e)" +%s)}
    export MUCHISIMOCKS_TRAIN_DEADLINE=$(( end - 45*60 )); echo "train deadline: $(date -d @${MUCHISIMOCKS_TRAIN_DEADLINE})"
    python run_inference.py --config-train ${config_train}
    rc=$?
    if [ $rc -eq 3 ]; then  # sbi_model.EXIT_TRAINING_INCOMPLETE: checkpointed, resume in a new job
        task="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        echo "$(date): training incomplete; requeueing ${task}"
        scontrol requeue "${task}" && exit 0
        exit 3
    fi
    [ $rc -eq 0 ] || exit $rc
fi
for config_test in ${configs_test}; do
    echo "test: ${config_test}"
    # retries: /scicomp drops out now and then (ImportError libgsl.so.25 mid-job, outages up to ~40 min); sampling resumes
    ok=0
    for attempt in 1 2 3 4; do
        python run_inference.py --config-test ${config_test} && { ok=1; break; }
        [ ${attempt} -lt 4 ] && { echo "$(date): test failed (attempt ${attempt}), retrying in 15 min"; sleep 900; }
    done
    [ ${ok} -eq 1 ] || exit 1
done
echo "Done $(date)"
