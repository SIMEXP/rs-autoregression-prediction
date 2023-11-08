#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --job-name=scaling
#SBATCH --output=/lustre04/scratch/hwang1/logs/%x_%A.out
#SBATCH --error=/lustre04/scratch/hwang1/logs/%x_%A.out
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

source /lustre03/project/6003287/${USER}/.virtualenvs/rs-autoregression-prediction/bin/activate
# print the amount of memory requested
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE}"
cd /lustre03/project/6003287/${USER}/rs-autoregression-prediction
python src/scaling.py --multirun ++experiment.scaling.segment=1 \
    ++experiment.scaling.n_sample=-1
