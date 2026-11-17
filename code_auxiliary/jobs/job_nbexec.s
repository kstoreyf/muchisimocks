#!/bin/bash
# Execute a notebook on a compute node; executed copy -> /scratch/kstoreyf/muchisimocks/nbexec/<name>_executed.ipynb
# (inspect, then install over the source atomically). Usage: sbatch --job-name=nbexec_<x> jobs/job_nbexec.s notebooks/<nb>.ipynb
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/scratch/kstoreyf/muchisimocks/logs_v2/%x-%j.out
. ~/load_modules.sh
source /scicomp/builds/Rocky/8.7/Common/software/Anaconda3/2023.03-1/etc/profile.d/conda.sh
conda activate benv
NB=$(realpath $1)
OUTDIR=/scratch/kstoreyf/muchisimocks/nbexec
mkdir -p $OUTDIR
cd $(dirname $NB)
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3000 --output-dir $OUTDIR --output $(basename $NB .ipynb)_executed.ipynb $NB && echo NBEXEC_OK
