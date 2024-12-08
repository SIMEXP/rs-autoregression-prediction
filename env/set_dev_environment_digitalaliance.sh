# set up environment on compute canada
# install on the log in node
# Install in a virtual environment
# virtualenv -p /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/python/3.11/bin/python -a . env_py310
#!/bin/bash

TORCH=2.3.0
CUDA=cu117  # options: cpu, cu116, cu117

pip install --upgrade pip
pip install --no-index lightning
pip install --no-index pyg_lib torch-scatter torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
pip install --no-index -r env/requirements.txt
pip install hydra-submitit-launcher==1.2.0 nilearn==0.9.2
pip install git+https://github.com/SIMEXP/general_class_balancer.git
pip install git+https://github.com/Epistimio/orion.git@v0.2.7
pip install git+https://github.com/Epistimio/sample-space.git@v1.0.0
pip install configspace hydra-orion-sweeper==1.6.4
pip install -e fmri_autoreg
pip install -e .
# style related things below. Feel free to ignore
module load rust
pip install -r env/requirements-dev.txt
pre-commit install

# mount back the new update
module load gentoo/2023
