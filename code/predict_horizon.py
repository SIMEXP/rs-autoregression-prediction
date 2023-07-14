import argparse
import csv
import json
import os
import pickle as pk
import re
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import torch
from src.data.load_data import load_params
from src.models.predict_model import predict_horizon
from src.tools import check_path, load_model

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

    model_path = (
        args.model if args.model.exists() else args.model / "model.pkl"
    )
    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()

    # generate feature (prediction of n+1) using all the data
    batch_size = params["batch_size"] if "batch_size" in params else 100
    r2, Z, Y = predict_horizon(
        model=model,
        seq_length=params["seq_length"],
        horizon=HORIZON,
        data_file=params["data_file"],
        task_filter=params["data_filter"],
        batch_size=batch_size,
        stride=params["time_stride"],
        standardize=False,
    )
    np.save(os.path.join(output_dir, f"r2_horizon-{HORIZON}.npy"), r2)
    np.save(os.path.join(output_dir, f"Z_horizon-{HORIZON}.npy"), Z)
    np.save(os.path.join(output_dir, f"Y_horizon-{HORIZON}.npy"), Y)


if __name__ == "__main__":
    main()
