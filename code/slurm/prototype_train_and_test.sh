#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --output=/lustre04/scratch/hwang1/logs/%x_%A.out
#SBATCH --error=/lustre04/scratch/hwang1/logs/%x_%A.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

source /lustre03/project/6003287/${USER}/.virtualenvs/rs-autoregression-prediction/bin/activate
cd /lustre03/project/6003287/${USER}/rs-autoregression-prediction

python "code/train_and_test.py" \
    -o outputs/predict_horizon \
    -p code/parameters/prototype.json \
    -v 1
