# set up environment on compute canada
# install on the log in node
# Install in a virtual environment
# virtualenv -p /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx512/Core/python/3.10.2/bin/python -a . env_py310
#!/bin/bash

TORCH=1.13.1
CUDA=cu117  # options: cpu, cu116, cu117

pip install --upgrade pip
pip install --no-index torch==${TORCH}
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install --no-index -r env/requirements.txt
pip install git+https://github.com/Epistimio/orion.git@v0.2.7
pip install git+https://github.com/Epistimio/sample-space.git@v1.0.0
pip install nilearn==0.9.2 hydra-orion-sweeper==1.6.4 hydra-submitit-launcher==1.2.0
pip install -e fmri_autoreg
pip install -e .
# style related things below. Feel free to ignore
module load rust
pip install -r env/requirements-dev.txt
pre-commit install
