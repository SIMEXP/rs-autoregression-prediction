# set up environment on compute canada
# install on the log in node
#!/bin/bash

TORCH=1.13.1
CUDA=cpu

pip install --upgrade pip
pip install --no-index torch==${TORCH}
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
pip install -r code/fmri-autoreg/requirements.txt
pip install -e code/fmri-autoreg
pip install -r env/requirements-dev.txt
pre-commit install
