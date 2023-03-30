# set up environment on compute canada
# install on the log in node
#!/bin/bash

pip install --no-index torch==1.13.1
pip install pyg-lib
pip install --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
pip install -r code/fmri-autoreg/requirements.txt
pip install -e code/fmri-autoreg
