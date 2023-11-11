python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=10 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=4 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=50,100,200,300,400,500 \
  ++random_state=1,2,3,12,666,999,728,503,42

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=30 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=8 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=600,700,800,900,1000 \
  ++random_state=1,2,3,12,666,999,728,503,42

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=120 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=64 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=2000,3000,4000,5000,6000,7000,8000,9000 \
  ++random_state=1,2,3,12,666,999,728,503,42

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=240 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=128 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,-1 \
  ++random_state=1,2,3,12,666,999,728,503,42
