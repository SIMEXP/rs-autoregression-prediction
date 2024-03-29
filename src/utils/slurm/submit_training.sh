python src/train.py --multirun hydra=hyperparameters \
  ++hydra.launcher.timeout_min=60 \
  ++hydra.launcher.mem_gb=4 \
  ++data.n_sample=1000 \
  ++torch_device=cuda:0

python src/train.py \
  ++data.n_sample=1000 \
  ++torch_device=cpu \
  ++model.batch_size=127 \
  ++model.lag=1 \
  ++model.lr=0.002561 \
  ++model.lr_thres=3.01e-10 \
  ++model.seq_length=30

# train small default model with the full ukbb
python src/train.py --multirun hydra=hyperparameters \
  ++hydra.launcher.timeout_min=240 \
  ++hydra.launcher.mem_gb=4 \
  ++torch_device=cpu \
  ++data.n_sample=-1

python src/train.py --multirun  \
  hydra/launcher=submitit_slurm \
  ++hydra.launcher.account=rrg-pbellec \
  ++hydra.launcher.timeout_min=90 \
  ++hydra.launcher.mem_gb=4 \
  ++hydra.launcher.gpus_per_node=1 \
  ++data.n_sample=-1
