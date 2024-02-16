import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from fmri_autoreg.models.predict_model import predict_horizon
from fmri_autoreg.tools import load_model
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="extract")
def main(params: DictConfig) -> None:
    """Train model using parameters dict and save results."""

    from src.data.load_data import load_data
    from src.model.extract_features import extract_convlayers

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Working on {device}.")

    compute_edge_index = "Chebnet" in params["model"]["model"]
    thres = params["data"]["edge_index_thres"] if compute_edge_index else None
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")

    model_path = Path(params["model_path"])

    output_conv_path = Path(output_dir) / "feature_convlayers.h5"
    output_horizon_path = (
        Path(output_dir) / f"feature_horizon-{params['horizon']}.h5"
    )

    # load test set subject path from the training
    with open(model_path.parent / "train_test_split.json", "r") as f:
        subj = json.load(f)

    data = subj["test"]

    # save test data path to a text file for easy future reference
    with open(output_dir / "test_set_connectome.txt", "w") as f:
        for item in data:
            f.write("%s\n" % item)

    # save the model parameters in the h5 files
    for path in [output_conv_path, output_horizon_path]:
        with h5py.File(path, "a") as f:
            f.attrs["complied_date"] = str(datetime.today())
            f.attrs["based_on_model"] = str(model_path)
            if "horizon" in path.name:
                f.attrs["horizon"] = params["horizon"]

    # generate feature for each subject
    log.info("Load model")
    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()
    log.info(f"Predicting t+{params['horizon']} of each subject")
    for h5_dset_path in tqdm(data):
        # get the prediction of t+1
        r2, Z, Y = predict_horizon(
            model=model,
            seq_length=params["model"]["seq_length"],
            horizon=params["horizon"],
            data_file=params["data"]["data_file"],
            dset_path=h5_dset_path,
            batch_size=params["data"]["batch_size"],
            stride=params["data"]["time_stride"],
            standardize=False,  # the ts is already standardized
        )
        # save the original output to a h5 file
        with h5py.File(output_horizon_path, "a") as f:
            for value, key in zip([r2, Z, Y], ["r2map", "Z", "Y"]):
                new_ds_path = h5_dset_path.replace("timeseries", key)
                f[new_ds_path] = value

    log.info("extract convo layers")
    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()
    for h5_dset_path in tqdm(data):
        convlayers, convlayers_F = extract_convlayers(
            data_file=params["data"]["data_file"],
            h5_dset_path=h5_dset_path,
            model=model,
            seq_length=params["model"]["seq_length"],
            time_stride=params["data"]["time_stride"],
            lag=params["data"]["lag"],
            compute_edge_index=compute_edge_index,
            thres=thres,
            device=torch.device(device),
        )
        # save the original output to a h5 file
        with h5py.File(output_conv_path, "a") as f:
            new_ds_path = h5_dset_path.replace("timeseries", "convlayers")
            f[new_ds_path] = convlayers
    # save the original output to a h5 file
    with h5py.File(output_conv_path, "a") as f:
        f.attrs["convolution_layers_F"] = convlayers_F


if __name__ == "__main__":
    main()
