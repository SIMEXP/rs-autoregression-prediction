import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from data.load_data import load_data
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from model.extract_features import extract_convlayers
from omegaconf import DictConfig, OmegaConf
from src.models.predict_model import predict_horizon
from src.tools import check_path, load_model
from tqdm import tqdm


def save_h5(
    data_file: Path,
    h5_output_path: Path,
    h5_dset_path: str,
    features: list,
    features_names: list,
    phenotypes: List[str] = ("sex", "age", "diagnosis"),
):
    site_dataset_name = h5_dset_path.split("/")[1]
    with h5py.File(h5_output_path, "a") as f:
        for value, key in zip(features, features_names):
            new_ds_path = h5_dset_path.replace("timeseries", key)
            f[new_ds_path] = value
        # save pneotype info as attributes of the subject
        subject_path = ("/").join(new_ds_path.split("/")[:-1])
        f[subject_path].attrs["site"] = site_dataset_name
        for att in phenotypes:
            f[subject_path].attrs[att] = load_data(
                data_file, h5_dset_path, dtype=att
            )[0]


def r2_stats(
    h5_dset_path,
    data_file,
    r2,
    phenotypes: List[str] = ("sex", "age", "diagnosis"),
):
    site_dataset_name = h5_dset_path.split("/")[1]
    scan_identifier = h5_dset_path.split("/")[-1].split("_atlas")[0]
    summary_stats = {
        "r2mean": [r2.mean()],
        "site": [site_dataset_name],
    }
    for pheno in phenotypes:
        summary_stats[pheno] = load_data(data_file, h5_dset_path, dtype=pheno)
    summary_stats = pd.DataFrame(summary_stats)
    summary_stats.index = [scan_identifier]
    return summary_stats


@hydra.main(version_base="1.3", config_path="../config", config_name="extract")
def main(params: DictConfig) -> None:
    """Train model using parameters dict and save results."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Working on {device}.")

    compute_edge_index = params["model"]["model"] == "Chebnet"
    thres = params["data"]["edge_index_thres"] if compute_edge_index else None
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Current working directory : {os.getcwd()}")
    print(f"Output directory  : {output_dir}")

    model_path = params["model_path"]

    output_conv_path = output_dir / "feature_convlayers.h5"
    output_horizon_path = (
        output_dir / f"feature_horizon-{params['horizon']}.h5"
    )
    output_stats_path = (
        output_dir / f"figures/feature_horizon-{params['horizon']}.tsv"
    )
    (Path(output_dir) / "figures").mkdir(exist_ok=True)

    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()

    # load subject path from the training
    with open(model_path.parent / "train_test_split.json", "r") as f:
        subj = json.load(f)

    data = []
    for s in subj.values():
        data += s

    # save the model parameters in the h5 files
    for path in [output_conv_path, output_horizon_path]:
        with h5py.File(path, "a") as f:
            f.attrs["complied_date"] = str(datetime.today())
            f.attrs["based_on_model"] = str(model_path)
            if "horizon" in path.name:
                f.attrs["horizon"] = params["horizon"]

    df_for_stats = []

    # generate feature for each subject
    print("Predicting t+1 of each subject and extract convo layers")
    for h5_dset_path in tqdm(data):
        # get the prediction of t+1
        r2, Z, Y = predict_horizon(
            model=model,
            seq_length=params["model"]["seq_length"],
            horizon=params["horizon"],
            data_file=params["data"]["data_file"],
            task_filter=h5_dset_path,
            batch_size=params["data"]["batch_size"],
            stride=params["data"]["time_stride"],
            standardize=False,
        )

        # save mean r2 in a dataframe
        df_for_stats.append(
            r2_stats(
                h5_dset_path,
                params["data"]["data_file"],
                r2,
                ["sex", "age", "diagnosis"],
            )
        )

        # save the original output to a h5 file
        save_h5(
            params["data"]["data_file"],
            output_horizon_path,
            h5_dset_path,
            [r2, Z, Y],
            ["r2map", "Z", "Y"],
            ["sex", "age", "diagnosis"],
        )

        convlayers = extract_convlayers(
            data_file=params["data"]["data_file"],
            h5_dset_path=h5_dset_path,
            model=model,
            seq_length=params["model"]["seq_length"],
            time_stride=params["data"]["time_stride"],
            lag=params["data"]["lag"],
            compute_edge_index=compute_edge_index,
            thres=thres,
        )
        # save the original output to a h5 file
        save_h5(
            params["data"]["data_file"],
            output_conv_path,
            h5_dset_path,
            [convlayers],
            ["convlayers"],
            ["sex", "age", "diagnosis"],
        )

    df_for_stats = pd.concat(df_for_stats)
    df_for_stats.to_csv(output_stats_path, sep="\t")


if __name__ == "__main__":
    main()
