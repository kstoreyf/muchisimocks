#!/bin/bash
# Submit: cd "$(dirname "$0")" && mkdir -p logs && sbatch "$0"
#SBATCH --qos=regular
#SBATCH --job-name=bispec_shame_p0_n1000_biasshame_noisebest_p0_n1_step100
##SBATCH --time=0:30:00  
##SBATCH --time=8:00:00 # time per task, but doing Nsteps; for 20000 (most), use 8h to be safe. lower, 1h fine
#SBATCH --time=24:00:00 # bispec takes a long time
#SBATCH --nodes=1              # nodes per instance
#SBATCH --cpus-per-task=1 # for compute_statistics.py
##SBATCH --cpus-per-task=24 # for make_quijote_matched_mocks.py
# was having issues with jobs failing, maybe due to 
# too many tasks submitted? with 100 at a time... also 50... careful! try 25
##x-y%z; start x, end y INCLUSIVE, z tasks at a time max
##(Y-X)*step_size = total you want to run
##SBATCH --array=0-99%40 # for 10000 training set
#SBATCH --array=0-9 # for 1000 test sets
##SBATCH --array=0-0
#SBATCH --mem=3G # 2G for bispectrum, 1G too low; 3G for pnn, 2G too low (??)
##SBATCH --mem=35G
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

# make quijote matched mocks
#idxs_LH=("0037" "0574" "0822" "1082" "1510" "0254" "0663" "0977" "1317" "1642")
#idx_LH=${idxs_LH[${SLURM_ARRAY_TASK_ID}]}
#python make_quijote_matched_mocks.py --idxs_LH ${idx_LH} --include_damping


### COMPUTE_STATISTICS.PY

### noiseless
# train
#python compute_statistics.py --statistic pnn --tag_params _p5_n10000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
#python compute_statistics.py --statistic pgm --tag_params _p5_n10000 --tag_biasparams _biasnest_p4_n320000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
#python compute_statistics.py --statistic bispec --tag_params _p5_n10000 --tag_biasparams _biasnest_p4_n320000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
# test
#python compute_statistics.py --statistic pnn --tag_params _coverage_p5_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic pgm --tag_params _coverage_p5_n1000 --tag_biasparams _biascoverage_p4_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic bispec --tag_params _coverage_p5_n1000 --tag_biasparams _biascoverage_p4_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
# fixed cosmo
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _quijote_p0_n1000 --tag_biasparams _b1000_p0_n1
#python compute_statistics.py --statistic pgm --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _quijote_p0_n1000 --tag_biasparams _b1000_p0_n1
#python compute_statistics.py --statistic pnn --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _shame_p0_n1000
#python compute_statistics.py --statistic pgm --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _shame_p0_n1000 --tag_biasparams _biasshame_p0_n1
#python compute_statistics.py --statistic bispec --tag_params _shame_p0_n1000 --tag_biasparams _biasshame_p0_n1 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}

### noisy anmult
# train
#python compute_statistics.py --statistic pk --tag_params _p5_n10000 --tag_biasparams _biasnoisenest_p9_n320000 --tag_noise _noise_unit_p5_n10000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
#python compute_statistics.py --statistic pgm --tag_params _p5_n10000 --tag_biasparams _biasnoisenest_p9_n320000 --tag_noise _noise_unit_p5_n10000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
#python compute_statistics.py --statistic bispec --tag_params _p5_n10000 --tag_biasparams _biasnoisenest_p9_n320000 --tag_noise _noise_unit_p5_n10000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
# coverage test
#python compute_statistics.py --statistic pk --tag_params _coverage_p5_n1000 --tag_biasparams _biasnoisecoverage_p9_n1000 --tag_noise _noise_unit_coverage_p5_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic pgm --tag_params _coverage_p5_n1000 --tag_biasparams _biasnoisecoverage_p9_n1000 --tag_noise _noise_unit_coverage_p5_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic bispec --tag_params _coverage_p5_n1000 --tag_biasparams _biasnoisecoverage_p9_n1000 --tag_noise _noise_unit_coverage_p5_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
# fixed cosmo test with fixed bias, best noise
#python compute_statistics.py --statistic pk --tag_params _shame_p0_n1000 --tag_biasparams _biasshame_noisebest_p0_n1 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic pgm --tag_params _shame_p0_n1000 --tag_biasparams _biasshame_noisebest_p0_n1 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
python compute_statistics.py --statistic bispec --tag_params _shame_p0_n1000 --tag_biasparams _biasshame_noisebest_p0_n1 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 

### noisy m2p3 noise model
# train
#python compute_statistics.py --statistic pk --tag_params _p5_n10000 --tag_biasparams _biasnoisem2nest_p7_n320000 --tag_noise _noise_unit_p5_n10000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
#python compute_statistics.py --statistic pgm --tag_params _p5_n10000 --tag_biasparams _biasnoisem2nest_p7_n320000 --tag_noise _noise_unit_p5_n10000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
#python compute_statistics.py --statistic bispec --tag_params _p5_n10000 --tag_biasparams _biasnoisem2nest_p7_n320000 --tag_noise _noise_unit_p5_n10000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end}
# coverage test m2p3
#python compute_statistics.py --statistic pk --tag_params _coverage_p5_n1000 --tag_biasparams _biasnoisem2coverage_p7_n1000 --tag_noise _noise_unit_coverage_p5_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic pgm --tag_params _coverage_p5_n1000 --tag_biasparams _biasnoisem2coverage_p7_n1000 --tag_noise _noise_unit_coverage_p5_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic bispec --tag_params _coverage_p5_n1000 --tag_biasparams _biasnoisem2coverage_p7_n1000 --tag_noise _noise_unit_coverage_p5_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
# fixed cosmo test with fixed bias, free noise
#python compute_statistics.py --statistic pk --tag_params _shame_p0_n1000 --tag_biasparams _bias_shame_noise_p5_n1000 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic pgm --tag_params _shame_p0_n1000 --tag_biasparams _bias_shame_noise_p5_n1000 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic bispec --tag_params _shame_p0_n1000 --tag_biasparams _bias_shame_noise_p5_n1000 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
# fixed cosmo test with fixed bias, free noise m2p3
#python compute_statistics.py --statistic pk --tag_params _shame_p0_n1000 --tag_biasparams _bias_shame_noisem2_p3_n1000 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic pgm --tag_params _shame_p0_n1000 --tag_biasparams _bias_shame_noisem2_p3_n1000 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic bispec --tag_params _shame_p0_n1000 --tag_biasparams _bias_shame_noisem2_p3_n1000 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
# fixed cosmo test with fixed bias, best noise m2p3
#python compute_statistics.py --statistic pk --tag_params _shame_p0_n1000 --tag_biasparams _biasshame_noisem2best_p0_n1 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic pgm --tag_params _shame_p0_n1000 --tag_biasparams _biasshame_noisem2best_p0_n1 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 
#python compute_statistics.py --statistic bispec --tag_params _shame_p0_n1000 --tag_biasparams _biasshame_noisem2best_p0_n1 --tag_noise _noise_unit_shame_p0_n1000 --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} 


###########################################################
# OLD 
###########################################################
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

### noisy - mult
# training
# 1x bias per cosmo
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _p5_n10000 --tag_biasparams _biaszen_p4_n10000 --tag_noise _noise_unit_p5_n10000 --tag_Anoise _Anmult_p2_n10000
# 20x bias per cosmo
#python compute_statistics.py --statistic pk --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _p5_n10000 --tag_biasparams _biaszen_p4_n200000 --tag_noise _noise_unit_p5_n10000 --tag_Anoise _Anmult_p2_n200000
#python compute_statistics.py --statistic pgm --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _p5_n10000 --tag_biasparams _biaszen_p4_n200000 --tag_noise _noise_unit_p5_n10000 --tag_Anoise _Anmult_p5_n200000
# testing
# CV quijote
#python compute_statistics.py --statistic pk --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _quijote_p0_n1000 --tag_biasparams _b1000_p0_n1 --tag_noise _noise_unit_quijote_p0_n1000 --tag_Anoise _Anmult_p0_n1
#python compute_statistics.py --statistic pgm --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _quijote_p0_n1000 --tag_biasparams _b1000_p0_n1 --tag_noise _noise_unit_quijote_p0_n1000 --tag_Anoise _Anmult_p0_n1
# coverage
#python compute_statistics.py --statistic bispec --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _test_p5_n1000 --tag_biasparams _biaszen_p4_n1000 --tag_noise _noise_unit_test_p5_n1000 --tag_Anoise _Anmult_p5_n1000
#python compute_statistics.py --statistic pgm --idx_mock_start ${idx_mock_start} --idx_mock_end ${idx_mock_end} --tag_params _test_p5_n1000 --tag_biasparams _biaszen_p4_n1000 --tag_noise _noise_unit_test_p5_n1000 --tag_Anoise _Anmult_p5_n1000