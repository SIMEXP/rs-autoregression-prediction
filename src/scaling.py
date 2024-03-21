import csv
import json
import logging
import os
import pickle as pk
import re
import sys
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import submitit
import torch
from fmri_autoreg.data.load_data import (
    get_edge_index,
    load_data,
    load_params,
    make_seq,
)
from fmri_autoreg.models.predict_model import predict_model
from fmri_autoreg.models.train_model import train
from hydra.utils import instantiate
from omegaconf import DictConfig
from seaborn import lineplot
from sklearn.metrics import r2_score

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3", config_path="../config", config_name="training_scaling"
)
def main(params: DictConfig) -> None:
    """Train model using parameters dict and save results."""
    # import local library here because sumbitit and hydra being weird

    env = submitit.JobEnvironment()
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Process ID {os.getpid()} in {env}")
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")

    # organise parameters
    compute_edge_index = "Chebnet" in params["model"]["model"]
    thres = params["data"]["edge_index_thres"] if compute_edge_index else None
    train_param = {**params["model"], **params["experiment"]}
    train_param["batch_size"] = params["data"]["batch_size"]
    train_param["data_file"] = params["data"]["data_file"]
    train_param["time_stride"] = params["data"]["time_stride"]
    train_param["lag"] = params["data"]["lag"]
    train_param["num_workers"] = params["data"]["num_workers"]
    log.info(f"Random seed {params['random_state']}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    params["torch_device"] = device
    log.info(params["model"]["model"])
    log.info(f"Working on {device}.")

    # load data path
    n_sample = params["experiment"]["scaling"]["n_sample"]
    data_reference = instantiate(params["experiment"]["scaling"])
    with open(Path(output_dir) / "train_test_split.json", "w") as f:
        json.dump(data_reference, f, indent=2)
    if n_sample == -1:
        n_sample = (
            len(data_reference["train"])
            + len(data_reference["val"])
            + len(data_reference["test"])
        )
    log.info(f"Experiment on {n_sample} subjects. Load data.")

    # create connectome from training set
    if compute_edge_index:
        edge_index = get_edge_index(
            data_file=params["data"]["data_file"],
            dset_paths=data_reference["train"],
            threshold=thres,
        )
    else:
        edge_index = None
    train_data = (data_reference["train"], data_reference["val"], edge_index)
    (
        model,
        mean_r2_tng,
        mean_r2_val,
        losses,
        _,
    ) = train(train_param, train_data, verbose=params["verbose"])
    # save training results
    np.save(os.path.join(output_dir, "mean_r2_tng.npy"), mean_r2_tng)
    np.save(os.path.join(output_dir, "mean_r2_tng.npy"), mean_r2_tng)
    np.save(os.path.join(output_dir, "training_losses.npy"), losses)
    log.info(f"Mean r2 tng: {mean_r2_tng}")
    log.info(f"Mean r2 val: {mean_r2_val}")

    model = model.to(device)
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pk.dump(model, f)

    # visualise training loss
    training_losses = pd.DataFrame(losses)
    plt.figure()
    g = lineplot(data=training_losses)
    g.set_title(f"Training Losses (N={n_sample})")
    g.set_xlabel("Epoc")
    g.set_ylabel("Loss (MSE)")
    plt.savefig(Path(output_dir) / "training_losses.png")

    # make test labels
    r2_test = predict_model(
        model=model,
        params=train_param,
        data_file=train_param["data_file"],
        dset=data_reference["test"],
    )
    mean_r2_test = np.mean(r2_test)
    log.info(f"Mean r2 test: {mean_r2_test}")

    # save predict results
    np.save(os.path.join(output_dir, "r2_test.npy"), r2_test)
    del r2_test


if __name__ == "__main__":
    main()
