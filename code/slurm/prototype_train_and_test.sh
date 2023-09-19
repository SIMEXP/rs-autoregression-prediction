#!/bin/bash
#SBATCH --account=rrg-pbellec
#SBATCH --output=/lustre04/scratch/hwang1/logs/%x_%A.out
#SBATCH --error=/lustre04/scratch/hwang1/logs/%x_%A.out
#SBATCH --time=6:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

source /lustre03/project/6003287/${USER}/.virtualenvs/rs-autoregression-prediction/bin/activate
cd /lustre03/project/6003287/${USER}/rs-autoregression-prediction

python "code/train_and_test.py" \
    -o outputs/prototype \
    -p code/parameters/prototype.json \
    -v 1

python "code/extract_features.py" \
    -o outputs/prototype/features \
    -m outputs/prototype/model.pkl \
    -p code/parameters/prototype.json \
    -v 1

python "code/baseline_dx_prediction.py" \
    -m outputs/prototype/ \
    -f outputs/prototype/features/
