import csv
import json
import logging
import os
import pickle as pk
import re
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.load_data import load_data, load_h5_data_path, split_data_by_site
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from seaborn import lineplot
from sklearn.metrics import r2_score
from src.data.load_data import load_params, make_input_labels, make_seq
from src.models.train_model import train

# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3", config_path="../config", config_name="training_scaling"
)
def main(params: DictConfig) -> None:
    """Train model using parameters dict and save results."""
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")

    # organise parameters
    compute_edge_index = params["model"]["model"] == "Chebnet"
    thres = params["data"]["edge_index_thres"] if compute_edge_index else None
    train_param = {**params["model"], **params["experiment"]}
    train_param["batch_size"] = params["data"]["batch_size"]

    # load data
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
    tng_data = load_data(
        params["data"]["data_file"], data_reference["train"], dtype="data"
    )
    val_data = load_data(
        params["data"]["data_file"], data_reference["val"], dtype="data"
    )
    test_data = load_data(
        params["data"]["data_file"], data_reference["test"], dtype="data"
    )

    # training data labels
    log.info("Create training data labels.")
    X_tng, Y_tng, X_val, Y_val, edge_index = make_input_labels(
        tng_data,
        val_data,
        params["model"]["seq_length"],
        params["data"]["time_stride"],
        params["data"]["lag"],
        compute_edge_index,
        thres,
    )
    train_data = (X_tng, Y_tng, X_val, Y_val, edge_index)
    del tng_data
    del val_data

    log.info("Training data labels.")
    # train model
    (
        model,
        r2_tng,
        r2_val,
        Z_tng,
        Y_tng,
        Z_val,
        Y_val,
        losses,
        _,
    ) = train(train_param, train_data, verbose=params["verbose"])

    # save training results
    np.save(os.path.join(output_dir, "r2_tng.npy"), r2_tng)
    np.save(os.path.join(output_dir, "r2_val.npy"), r2_val)
    np.save(os.path.join(output_dir, "pred_tng.npy"), Z_tng)
    np.save(os.path.join(output_dir, "labels_tng.npy"), Y_tng)
    np.save(os.path.join(output_dir, "pred_val.npy"), Z_val)
    np.save(os.path.join(output_dir, "labels_val.npy"), Y_val)
    np.save(os.path.join(output_dir, "training_losses.npy"), losses)

    mean_r2_val = np.mean(r2_val)
    mean_r2_tng = np.mean(r2_tng)
    log.info(f"Mean r2 tng: {mean_r2_tng}")
    log.info(f"Mean r2 val: {mean_r2_val}")

    del Z_tng
    del Y_tng
    del r2_tng
    del Z_val
    del Y_val
    del r2_val

    model = model.to("cpu")
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pk.dump(model, f)

    # make test labels
    X_test, Y_test = make_seq(
        test_data,
        params["model"]["seq_length"],
        params["data"]["time_stride"],
        params["data"]["lag"],
    )
    del test_data

    # predict on test data
    Z_test = np.concatenate(
        [
            model.predict(x)
            for x in np.array_split(
                X_test, ceil(X_test.shape[0] / params["data"]["batch_size"])
            )
        ]
    )
    r2_test = r2_score(Y_test, Z_test, multioutput="raw_values")

    mean_r2_test = np.mean(r2_test)
    log.info(f"Mean r2 test: {mean_r2_test}")

    # save predict results
    np.save(os.path.join(output_dir, "r2_test.npy"), r2_test)
    np.save(os.path.join(output_dir, "pred_test.npy"), Z_test)
    np.save(os.path.join(output_dir, "labels_test.npy"), Y_test)
    del Z_test
    del Y_test
    del r2_test

    # visualise training loss
    training_losses = pd.DataFrame(losses)
    plt.figure()
    g = lineplot(data=training_losses)
    g.set_title(f"Training Losses (N={n_sample})")
    g.set_xlabel("Epoc")
    g.set_ylabel("Loss (MSE)")
    plt.savefig(Path(output_dir) / "training_losses.png")


if __name__ == "__main__":
    main()
