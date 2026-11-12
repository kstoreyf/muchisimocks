#!/bin/bash
# Submit: cd "$(dirname "$0")" && mkdir -p logs && sbatch "$0"
#SBATCH --qos=regular
#SBATCH --job-name=aliasing_test
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x.out

echo "Current date and time: $(date)"
echo "Running on node ${SLURMD_NODENAME}"
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv

python test_deposit_aliasing.py
