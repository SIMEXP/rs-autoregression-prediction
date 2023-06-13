# set up environment
#!/bin/bash

pip install --upgrade pip
pip install torch==1.13.1
pip install pyg-lib
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
pip install -r code/fmri-autoreg/requirements.txt
pip install -e code/fmri-autoreg
pip install -r env/requirements-dev.txt
pre-commit install
