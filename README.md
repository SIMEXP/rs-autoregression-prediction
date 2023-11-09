# rs-autoregression-prediction

Autoregressive model on resting state time series benchmark

Based on https://github.com/FrancoisPgm/fmri-autoreg

## Dataset structure

- All inputs (i.e. building blocks from other sources) are located in
  `inputs/`.
- All custom code is located in `code/`.


## Compute Canada and environment set up

Python version 3.8.x.
The reference model we are using is code in pytorch `1.13.1`.

```
python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=30 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=8 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=100,200

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=10 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=4 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=50
```
