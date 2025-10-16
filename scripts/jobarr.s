#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=datagen_p5_n10000_vel_step100_round2
##SBATCH --job-name=datagen_fixedcosmo_step10
##SBATCH --job-name=datagen_fisher_quijote_step3
##SBATCH --job-name=datagen_test_p5_n1000_step10
##SBATCH --job-name=bispec_test_p5_n1000_step10
##SBATCH --job-name=bispec_p5_n10000_biaszen_p4_n200000_step100
##SBATCH --job-name=pk_noise_quijote_p0_n1000_step100
##SBATCH --job-name=pk_noise_p5_n10000_step100
##SBATCH --job-name=pk_noise_test_p5_n1000_step100
##SBATCH --job-name=pklin_quijote_p0_n1000_step100
##SBATCH --job-name=bispec_quijote_p0_n1000_b1000_p0_n1_step100
##SBATCH --job-name=bispec_quijote_p0_n1000_b1000_p0_n1_noise_quijote_p0_n1000_An1_p0_n1_step100
##SBATCH --job-name=bispec_test_p5_n1000_biaszen_p4_n1000_noise_test_p5_n1000_An_p1_n1000_step100
##SBATCH --job-name=bispec_p5_n10000_biaszen_p4_n200000_noise_p5_n10000_An1_p0_n1_step100
##SBATCH --time=0:10:00 # time per task, but doing Nsteps; ~10s for bispec 
##SBATCH --time=8:00:00 # time per task, but doing Nsteps; for 20000 (most), use 8h to be safe. lower, 1h fine
#SBATCH --time=24:00:00 #datagen
#SBATCH --nodes=1              # nodes per instance
#SBATCH --gres=gpu:1  #gpu for datagen; off for bispec
#SBATCH --cpus-per-task=1
##SBATCH --cpus-per-task=24
##SBATCH --ntasks=1             # tasks per instance
# was having issues with jobs failing, maybe due to 
# too many tasks submitted? with 100 at a time... careful! try 25
##x-y%z; start x, end y INCLUSIVE, z tasks at a time max
##(Y-X)*step_size = total you want to run
##SBATCH --array=0-99%20 # for 10000 training set
#SBATCH --array=0-99%5
##SBATCH --array=0-0
##SBATCH --array=0-9 # for 1000 test sets
##SBATCH --array=0-2%3
##SBATCH --array=99-99
#SBATCH --mem=35G # got OOM for 30 for datagen	     
##SBATCH --mem=2G # 2G for bispectrum, 1G too low
#SBATCH --output=logs/%x-%a.out


echo "Current date and time: $(date)"
echo "Slurm job id is ${SLURM_JOB_ID}"
echo "Running on node ${SLURMD_NODENAME}"
echo "SLURM_CPUS_ON_NODE = ${SLURM_CPUS_ON_NODE}"
echo "SLURM_JOB_NUM_NODES = ${SLURM_JOB_NUM_NODES}"
# https://hpc.nmsu.edu/discovery/slurm/job-arrays/
echo "Array job id is ${SLURM_ARRAY_JOB_ID}" # SLURM_JOB_ID + SLURM_ARRAY_TASK_ID
echo "Instance index is ${SLURM_ARRAY_TASK_ID}."
. ~/load_modules.sh
# via https://stackoverflow.com/a/65183109
#source ~/anaconda3/etc/profile.d/conda.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
#i=$((SLURM_ARRAY_TASK_ID-SLURM_ARRAY_TASK_MIN))
# changed this so it won't shift to start at zeri
i=${SLURM_ARRAY_TASK_ID}
step_size=100
echo "i=${i}"
#idx_mock_start=$((SLURM_ARRAY_TASK_MIN + i*step_size))
idx_mock_start=$((i*step_size))
idx_mock_end=$((idx_mock_start + step_size))
echo "idx_mock_start=${idx_mock_start}, idx_mock_end=${idx_mock_end}"

### DATA_CREATION_PIPELINE.PY

python data_creation_pipeline.py ${idx_mock_start} ${idx_mock_end} --tag_params '_p5_n10000'
#python data_creation_pipeline.py ${idx_mock_start} ${idx_mock_end} --modecosmo fixed
#python data_creation_pipeline.py ${idx_mock_start} ${idx_mock_end} --modecosmo fisher --tag_params='_fisher_quijote'
#python cuda_minimal.py


### COMPUTE_STATISTICS.PY

### noiseless
# train
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _p5_n10000 --tag_biasparams _biaszen_p4_n200000 
# test
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _quijote_p0_n1000
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _test_p5_n1000 --tag_biasparams _biaszen_p4_n1000 
# CV quijote
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _quijote_p0_n1000 --tag_biasparams _b1000_p0_n1

### noise-only
#python compute_statistics.py --statistic pk --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_noise _noise_p5_n10000
#python compute_statistics.py --statistic pk --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_noise _noise_quijote_p0_n1000
#python compute_statistics.py --statistic pk --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_noise _noise_test_p5_n1000

### noisy
# training
# span noise range
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _p5_n10000 --tag_biasparams _biaszen_p4_n200000 --tag_noise _noise_p5_n10000 --tag_Anoise _An_p1_n10000
# noise An=1
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _p5_n10000 --tag_biasparams _biaszen_p4_n200000 --tag_noise _noise_p5_n10000 --tag_Anoise _An1_p0_n1
# testing
# CV quijote
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _quijote_p0_n1000 --tag_biasparams _b1000_p0_n1 --tag_noise _noise_quijote_p0_n1000 --tag_Anoise _An1_p0_n1
# coverage
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _test_p5_n1000 --tag_biasparams _biaszen_p4_n1000 --tag_noise _noise_test_p5_n1000 --tag_Anoise _An_p1_n1000
