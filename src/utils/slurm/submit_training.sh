# hyperparameter
python src/train.py --multirun hydra=hyperparameters

# debug
python src/train.py --multirun hydra=scaling \
  ++hydra.launcher.timeout_min=180 \
  ++data.n_sample=-1 \
  ++model.FK=\'128,32,128,32,128,32,128,32\' \
  ++model.M=\'32,16,8,1\' \
  ++model.batch_size=256 \
  ++model.lag=1 \
  ++model.lr=0.04966 \
  ++model.lr_thres=0.4105 \
  ++model.dropout=0.02249 \
  ++model.seq_length=29

# scaling
python src/full_experiment.py --multirun  \
  ++hydra.launcher.mem_gb=8 \
  ++data.n_sample=16000,20000,-1 \
  ++random_state=0,1,2,4,8,42
  # ++data.n_sample=100,250,500,1000,2000,3000,4000,5000,6000,8000,10000,16000,20000,-1 \
  # ++random_state=0,1,2,4,8,42

# extraction - create symlink to model
# outputs/ccn2024/model/ -> to training results
python src/extract.py --multirun model_path=outputs/ccn2024/model/model.pkl
# outputs/ccn2024/extract/ -> to extraction results from outputs/ccn2024/model/
python src/predict.py --multirun model_path=outputs/ccn2024/extract

# one script for all scaling
python src/full_experiment.py --multirun  \
  ++hydra.launcher.timeout_min=600 \
  ++data.n_sample=-1,10000,5000,2500,1250,625,300,150,75 \
  ++random_state=0,1,2,4,8,42
