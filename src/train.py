import csv
import json
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
from seaborn import boxplot, lineplot
from sklearn.metrics import r2_score
from src.data.load_data import load_params, make_input_labels, make_seq
from src.models.predict_model import predict_horizon
from src.models.train_model import train
from src.tools import check_path


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(params: DictConfig) -> None:
    """Train model using parameters dict and save results."""
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Current working directory : {os.getcwd()}")
    print(f"Output directory  : {output_dir}")

    batch_size = (
        params["data"]["batch_size"] if "batch_size" in params else 100
    )
    compute_edge_index = params["model"]["model"] == "Chebnet"
    thres = params["data"]["edge_index_thres"] if compute_edge_index else None
    # load data
    tng_dset, val_dset = instantiate(params["data"]["training"])
    test_dset = instantiate(params["data"]["testing"])

    data_reference = {
        "train": tng_dset,
        "validation": val_dset,
        "test": test_dset,
    }
    with open(Path(output_dir) / "train_test_split.json", "w") as f:
        json.dump(data_reference, f, indent=2)

    tng_data = load_data(params["data"]["data_file"], tng_dset, dtype="data")
    val_data = load_data(params["data"]["data_file"], val_dset, dtype="data")
    test_data = load_data(params["data"]["data_file"], test_dset, dtype="data")

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
    train_param = {**params["model"], **params["experiment"]}
    train_param["batch_size"] = params["data"]["batch_size"]
    del tng_data
    del val_data
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
    np.save(os.path.join(output_dir, "r2_tng.npy"), r2_tng)
    np.save(os.path.join(output_dir, "r2_val.npy"), r2_val)
    np.save(os.path.join(output_dir, "pred_tng.npy"), Z_tng)
    np.save(os.path.join(output_dir, "labels_tng.npy"), Y_tng)
    np.save(os.path.join(output_dir, "pred_val.npy"), Z_val)
    np.save(os.path.join(output_dir, "labels_val.npy"), Y_val)
    np.save(os.path.join(output_dir, "training_losses.npy"), losses)

    mean_r2_val = np.mean(r2_val)
    mean_r2_tng = np.mean(r2_tng)

    del Z_tng
    del Y_tng
    del r2_tng
    del Z_val
    del Y_val
    del r2_val

    model = model.to("cpu")
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pk.dump(model, f)

    print("Predicting test set")
    X_test, Y_test = make_seq(
        test_data,
        params["model"]["seq_length"],
        params["data"]["time_stride"],
        params["data"]["lag"],
    )
    del test_data

    batch_size = (
        len(X_tng)
        if "batch_size" not in params["data"]
        else params["data"]["batch_size"]
    )
    Z_test = np.concatenate(
        [
            model.predict(x)
            for x in np.array_split(X_test, ceil(X_test.shape[0] / batch_size))
        ]
    )
    r2_test = r2_score(Y_test, Z_test, multioutput="raw_values")
    mean_r2_test = np.mean(r2_test)

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
    g.set_xlabel("Epoc")
    g.set_ylabel("Loss (MSE)")
    for fold, r2 in zip(
        ["tng", "val", "test"], [mean_r2_tng, mean_r2_val, mean_r2_test]
    ):
        print(f"r2 {fold}: {r2}")
    plt.savefig(Path(output_dir) / "training_losses.png")


if __name__ == "__main__":
    main()
