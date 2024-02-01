python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=60 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=4 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=50,100,200,300,400,500 \
  ++model.FK=\"128,32,128,32,128,32,128,32,128,32,64,16,64,16\" \
  ++model.M=\"128,64,32,16,8,1\" \
  ++model.dropout=0.1 \
  ++random_state=1,2,3,12,666,999,728,503,42 &

workon rs-autoregression-prediction-gpu-py38
python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=600 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=8 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=600,700,800,900,1000 \
  ++model.FK=\"128,32,128,32,128,32,128,32,128,32,64,16,64,16\" \
  ++model.M=\"128,64,32,16,8,1\" \
  ++model.dropout=0.1 \
  ++random_state=1,2,3,12,666,999,728,503,42 &

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=3120 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=128 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=20000,-1 \
  ++model.FK=\"128,32,128,32,128,32,128,32,128,32,64,16,64,16\" \
  ++model.M=\"128,64,32,16,8,1\" \
  ++model.dropout=0.1 \
  ++random_state=1,2,3,12,666,999,728,503,42 &

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=1560 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=64 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=5000,6000,7000,8000,9000 \
  ++model.FK=\"128,32,128,32,128,32,128,32,128,32,64,16,64,16\" \
  ++model.M=\"128,64,32,16,8,1\" \
  ++model.dropout=0.1 \
  ++random_state=1,2,3,12,666,999,728,503,42 &

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=2040 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=80 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=10000,11000,12000 \
  ++model.FK=\"128,32,128,32,128,32,128,32,128,32,64,16,64,16\" \
  ++model.M=\"128,64,32,16,8,1\" \
  ++model.dropout=0.1 \
  ++random_state=1,2,3,12,666,999,728,503,42 &

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=2700 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=80 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=13000,14000,15000,16000,17000 \
  ++model.FK=\"128,32,128,32,128,32,128,32,128,32,64,16,64,16\" \
  ++model.M=\"128,64,32,16,8,1\" \
  ++model.dropout=0.1 \
  ++random_state=1,2,3,12,666,999,728,503,42 &

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=3200 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=128 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=18000,19000 \
  ++model.FK=\"128,32,128,32,128,32,128,32,128,32,64,16,64,16\" \
  ++model.M=\"128,64,32,16,8,1\" \
  ++model.dropout=0.1 \
  ++random_state=1,2,3,12,666,999,728,503,42 &

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=4200 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=128 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=20000,-1 \
  ++model.FK=\"128,32,128,32,128,32,128,32,128,32,64,16,64,16\" \
  ++model.M=\"128,64,32,16,8,1\" \
  ++model.dropout=0.1 \
  ++random_state=1,2,3,12,666,999,728,503,42 &


python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=10 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=8 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=50,100,200,300,400,500 \
  ++random_state=1,2,3,12,666,999,728,503,42 &

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=120 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=16 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=1000,2000,3000,4000,5000,6000,7000,8000,9000,10000 \
  ++random_state=1,2,3,12,666,999,728,503,42 &

python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=150 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=64 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=11000,12000,13000,14000,15000,16000,17000,18000,19000 \
  ++random_state=1,2,3,12,666,999,728,503,42 &


python src/scaling.py \
  --multirun \
  hydra/launcher=submitit_slurm \
  hydra.launcher.timeout_min=180 \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.mem_gb=128 \
  hydra.launcher.account=rrg-pbellec \
  ++experiment.scaling.segment=1 \
  ++experiment.scaling.n_sample=20000,21000,22000,23000,24000,-1 \
  ++random_state=1,2,3,12,666,999,728,503,42 &
