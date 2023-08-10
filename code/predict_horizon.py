import argparse
import csv
import json
import os
import pickle as pk
import re
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from giga_companion.load_data import load_data
from src.data.load_data import load_params
from src.models.predict_model import predict_horizon
from src.tools import check_path, load_model
from tqdm import tqdm

HORIZON = 1


def main():
    """Train model using parameters dict and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", "-o", type=Path, help="output directory"
    )
    parser.add_argument(
        "--model", "-m", type=Path, help="Path to model file or dir."
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Working on {device}.")

    params = load_params(args.param)
    output_dir = args.output_dir
    output_dir = check_path(output_dir)
    os.makedirs(output_dir)
    output_dir = Path(output_dir)

    model_path = (
        args.model if args.model.exists() else args.model / "model.pkl"
    )
    model_name = model_path.parent.name

    output_metric_path = output_dir / f"{model_name}_horizon-{HORIZON}.h5"
    output_stats_path = output_dir / f"{model_name}_horizon-{HORIZON}.tsv"
    print(f"Save metrics to {str(output_metric_path)}")

    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()

    # load subject path from the training
    with open(args.model.parent / "train_test_split.json", "r") as f:
        subj = json.load(f)

    data = []
    for s in subj.values():
        data += s

    # generate feature (prediction of n+1) for each subject
    batch_size = params["batch_size"] if "batch_size" in params else 100
    with h5py.File(output_metric_path, "a") as f:
        f.attrs["complied_date"] = str(datetime.today())
        f.attrs["based_on_model"] = model_name

    print("Predicting t+1 of each subject.")
    df_for_stats = pd.DataFrame()
    for h5_dset_path in tqdm(data):
        site_dataset_name = h5_dset_path.split("/")[1]
        h5_dset_name = h5_dset_path.split("/")[-1]
        scan_identifier = h5_dset_name.split("_atlas")[0]

        r2, Z, Y = predict_horizon(
            model=model,
            seq_length=params["seq_length"],
            horizon=HORIZON,
            data_file=params["data_file"],
            task_filter=h5_dset_path,
            batch_size=batch_size,
            stride=params["time_stride"],
            standardize=False,
        )
        # save the original output to a h5 file
        with h5py.File(output_metric_path, "a") as f:
            for value, key in zip([r2, Z, Y], ["r2map", "Z", "Y"]):
                horizon_path = h5_dset_path.replace("timeseries", key)
                f[horizon_path] = value
            # save pneotype info as attributes of the subject
            subject_path = ("/").join(horizon_path.split("/")[:-1])
            f[subject_path].attrs["r2mean"] = r2.mean().mean()
            f[subject_path].attrs["site"] = site_dataset_name
            for att in ["sex", "age", "diagnosis"]:
                f[subject_path].attrs[att] = load_data(
                    params["data_file"], h5_dset_path, dtype=att
                )[0]

        # save mean r2 in a dataframe
        df = pd.DataFrame(
            {
                "r2mean": [r2.mean()],
                "sex": load_data(
                    params["data_file"], h5_dset_path, dtype="sex"
                ),
                "age": load_data(
                    params["data_file"], h5_dset_path, dtype="age"
                ),
                "diagnosis": load_data(
                    params["data_file"], h5_dset_path, dtype="diagnosis"
                ),
                "site": [site_dataset_name],
            }
        )
        df.index = [scan_identifier]
        df_for_stats = pd.concat([df_for_stats, df])
    df_for_stats.to_csv(output_stats_path, sep="\t")


if __name__ == "__main__":
    main()
