# set up environment on compute canada
# install on the log in node
#!/bin/bash

TORCH=1.13.1
CUDA=cu117  # options: cpu, cu116, cu117

pip install --upgrade pip
pip install --no-index torch==${TORCH}
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index \
    torch-scatter==2.1.1+computecanada \
    torch-sparse==0.6.17+computecanada \
    torch-cluster==1.6.1+computecanada \
    torch-spline-conv==1.2.2+computecanada \
    torch-geometric==2.3.1+computecanada \
    numpy==1.23.0 \
    cython
# fmri-autoreg dependencies - updated 2023-11-07
pip install h5py==3.6.0 \
    nilearn==0.9.2 \
    tqdm==4.64.1 \
    darts==0.16.0 \
    pandas==1.3.0
pip install -e src/fmri_autoreg

module load rust
pip install -r env/requirements-dev.txt
pip install -r env/requirements.txt
pre-commit install
