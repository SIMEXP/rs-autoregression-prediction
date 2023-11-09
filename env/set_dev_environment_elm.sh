# set up environment
#!/bin/bash

TORCH=1.13.1
CUDA=cpu

pip install --upgrade pip setuptools wheel
pip install torch==${TORCH}+${CUDA}
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install -r code/fmri_autoreg/requirements.txt
pip install -e src/fmri_autoreg
pip install -r env/requirements-dev.txt
pip install -r env/requirements.txt
pre-commit install
