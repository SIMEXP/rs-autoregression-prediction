import argparse
import csv
import os
import pickle as pk
import re
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
from giga_companion.load_data import (
    load_data,
    load_h5_data_path,
    split_data_by_site,
)
from sklearn.metrics import r2_score
from src.data.load_data import load_params, make_input_labels, make_seq
from src.models.predict_model import predict_horizon
from src.models.train_model import train
from src.tools import check_path

HORIZON = 1


def main():
    """Train model using parameters dict and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", "-o", type=Path, help="output directory"
    )
    parser.add_argument(
        "--param",
        "-p",
        type=Path,
        help="Parameters : path to json file or dict",
    )
    parser.add_argument("-n", type=int, help="Number of subjects to use.")
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        help="Verbosity level, 0 to 2. Default is 1.",
    )
    args = parser.parse_args()

    params = load_params(args.param)
    batch_size = params["batch_size"] if "batch_size" in params else 100
    compute_edge_index = params["model"] == "Chebnet"
    thres = params["edge_index_thres"] if compute_edge_index else None
    output_dir = args.output_dir
    output_dir = check_path(output_dir)
    os.makedirs(output_dir)

    # load data
    with h5py.File(params["data_files"], "r") as f:
        datasets = f.attrs["dataset_list"].tolist()

    # ABIDE 1
    tng_dset, val_dset = split_data_by_site(
        params["data_files"],
        datasets=[d for d in datasets if "abide1" in d],
        test_set=0.2,
        split_type=None,
        data_filter="abide1.*/*/sub-.*desc-197.*timeseries",
    )

    # ABIDE 2
    test_dset = load_h5_data_path(
        params["data_files"],
        "abide2.*/*/sub-.*desc-197.*timeseries",
        shuffle=True,
    )

    tng_data = load_data(params["data_files"], tng_dset, dtype="data")
    val_data = load_data(params["data_files"], val_dset, dtype="data")
    test_data = load_data(params["data_files"], test_dset, dtype="data")

    X_tng, Y_tng, X_val, Y_val, edge_index = make_input_labels(
        tng_data,
        val_data,
        params["seq_length"],
        params["time_stride"],
        params["lag"],
        compute_edge_index,
        thres,
    )
    train_data = (X_tng, Y_tng, X_val, Y_val, edge_index)
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
    ) = train(params, train_data, verbose=args.verbose)
    np.save(os.path.join(output_dir, "r2_tng.npy"), r2_tng)
    np.save(os.path.join(output_dir, "r2_val.npy"), r2_val)
    np.save(os.path.join(output_dir, "pred_tng.npy"), Z_tng)
    np.save(os.path.join(output_dir, "labels_tng.npy"), Y_tng)
    np.save(os.path.join(output_dir, "pred_val.npy"), Z_val)
    np.save(os.path.join(output_dir, "labels_val.npy"), Y_val)
    np.save(os.path.join(output_dir, "training_losses.npy"), losses)

    model = model.to("cpu")
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pk.dump(model, f)

    params["r2_mean_tng"] = r2_tng.mean()
    params["r2_std_tng"] = r2_tng.std()

    X_test, Y_test = make_seq(
        test_data, params["seq_length"], params["time_stride"], params["lag"]
    )
    del test_data
    batch_size = (
        len(X_tng) if "batch_size" not in params else params["batch_size"]
    )
    Z_test = np.concatenate(
        [
            model.predict(x)
            for x in np.array_split(X_test, ceil(X_test.shape[0] / batch_size))
        ]
    )
    r2_test = r2_score(Y_test, Z_test, multioutput="raw_values")

    np.save(os.path.join(output_dir, "r2_test.npy"), r2_test)
    np.save(os.path.join(output_dir, "pred_test.npy"), Z_test)
    np.save(os.path.join(output_dir, "labels_test.npy"), Y_test)

    # generate feature using all the data
    r2, Z, Y = predict_horizon(
        model=model,
        seq_length=params["seq_length"],
        horizon=HORIZON,
        data_file=params["test_data_file"],
        task_filter=params["test_task_filter"],
        batch_size=batch_size,
        stride=1,
        standardize=False,
    )
    np.save(os.path.join(output_dir, f"r2_horizon-{HORIZON}.npy"), r2)
    np.save(os.path.join(output_dir, f"Z_horizon-{HORIZON}.npy"), Z)
    np.save(os.path.join(output_dir, f"Y_horizon-{HORIZON}.npy"), Y)


if __name__ == "__main__":
    main()
