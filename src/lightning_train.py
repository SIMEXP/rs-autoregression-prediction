import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from src.data.ukbb_datamodule import UKBBDataModule
from src.model.autoreg_module import AutoregModule
from src.model.models import Chebnet, get_edge_index_threshold

# test the data module
data = UKBBDataModule(
    connectome_file="inputs/connectomes/ukbb_libral_scrub_20240716_connectome.h5",
    phenotype_file="inputs/connectomes/ukbb_libral_scrub_20240716_phenotype.tsv",
    data_dir="data/",
    atlas=("MIST", 197),
    proportion_sample=1.0,
    timeseries_segment=0,
    timeseries_window_stride_lag=(16, 1, 1),
    train_val_test_split=(0.6, 0.2, 0.2),
    class_balance_confounds=[
        "site",
        "sex",
        "age",
        "mean_fd_raw",
        "proportion_kept",
    ],
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    random_state=1,
)
