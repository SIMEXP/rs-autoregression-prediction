# use a small set to make sure the parameter tuning is doing things
python src/train.py --multirun hydra=hyperparameters ++hydra.launcher.timeout_min=480 ++data.n_sample=-1

# debug
python src/train.py \
  ++data.n_sample=-1 \
  ++model.FK=\'8,3,8,3,8,3\' \
  ++model.M=\'8,1\' \
  ++model.batch_size=127 \
  ++model.lag=3 \
  ++model.lr=0.65 \
  ++model.lr_thres=0.702 \
  ++model.seq_length=52

# train small default model with the full ukbb
python src/train.py --multirun hydra=hyperparameters \
  ++hydra.launcher.timeout_min=240 \
  ++hydra.launcher.mem_gb=4 \
  ++torch_device=cpu \
  ++data.n_sample=-1

# scaling
python src/train.py --multirun  \
  hydra=scaling \
  model=ccn_abstract \
  ++data.n_sample=100,250,500,1000,2000,3000,4000,5000,6000,8000,10000,16000,20000,-1 \
  ++random_state=0,1,2,4,8,42

# extraction - create symlink to model
python src/extract.py --multirun model_path=outputs/ccn2024/best_model/model.pkl
