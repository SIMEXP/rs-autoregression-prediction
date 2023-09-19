import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from giga_companion.load_data import load_data
from seaborn import boxplot
from src.data.load_data import load_params, make_input_labels
from src.models.predict_model import predict_horizon
from src.tools import check_path, load_model
from torch_geometric.nn import ChebConv
from tqdm import tqdm

HORIZON = 1


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        layer_activations = module_out
        self.outputs.append(layer_activations.detach().clone())

    def clear(self):
        self.outputs = []


def module_output_to_numpy(tensor):
    return tensor.detach().to("cpu").numpy()


def extract_convlayers(
    data_file,
    h5_dset_path,
    model,
    seq_length,
    time_stride,
    lag,
    compute_edge_index,
    thres,
):
    # load data
    ts = load_data(data_file, h5_dset_path, dtype="data")
    X_ts = make_input_labels(
        ts,
        [],
        seq_length,
        time_stride,
        lag,
        compute_edge_index,
        thres,
    )[
        0
    ]  # just one subject and the X

    # register the hooks to the pretrain model
    save_output = SaveOutput()
    hook_handles = []
    for _, module in model.named_modules():
        if isinstance(module, ChebConv):
            handle = module.register_forward_hook(save_output)
            hook_handles.append(handle)

    # pass the data through pretrained model
    _ = model(torch.tensor(X_ts))
    conv_layers = np.array(
        [module_output_to_numpy(o) for o in save_output.outputs]
    )  # get all layers (layer, batch, node, feature F)
    # first layer is nodes, since the rest will be compressed
    # (node, batch, layer, feature F)
    conv_layers = np.swapaxes(conv_layers, 0, 2)
    # remove the hooks
    for handle in hook_handles:
        handle.remove()
    return conv_layers


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
    compute_edge_index = params["model"] == "Chebnet"
    thres = params["edge_index_thres"] if compute_edge_index else None
    output_dir = args.output_dir
    output_dir = check_path(output_dir)
    os.makedirs(output_dir)
    output_dir = Path(output_dir)
    print(f"Save metrics to {str(output_dir)}")

    model_path = (
        args.model if args.model.exists() else args.model / "model.pkl"
    )
    model_name = model_path.parent.name

    output_conv_path = output_dir / f"{model_name}_convlayers.h5"
    output_horizon_path = output_dir / f"{model_name}_horizon-{HORIZON}.h5"
    output_stats_path = (
        output_dir / f"figures/{model_name}_horizon-{HORIZON}.tsv"
    )
    (Path(output_dir) / "figures").mkdir(exist_ok=True)

    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()

    # load subject path from the training
    with open(args.model.parent / "train_test_split.json", "r") as f:
        subj = json.load(f)

    data = []
    for s in subj.values():
        data += s

    # save the model parameters in the h5 files
    batch_size = params["batch_size"] if "batch_size" in params else 100
    for path in [output_conv_path, output_horizon_path]:
        with h5py.File(path, "a") as f:
            f.attrs["complied_date"] = str(datetime.today())
            f.attrs["based_on_model"] = model_name
            if "horizon" in path.name:
                f.attrs["horizon"] = HORIZON
    df_for_stats = pd.DataFrame()

    # generate feature for each subject
    print("Predicting t+1 of each subject and extract convo layers")
    for h5_dset_path in tqdm(data):
        # get the prediction of t+1
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
        with h5py.File(output_horizon_path, "a") as f:
            for value, key in zip(
                [r2, Z, Y],
                ["r2map", "Z", "Y"],
            ):
                new_ds_path = h5_dset_path.replace("timeseries", key)
                f[new_ds_path] = value
            # save pneotype info as attributes of the subject
            subject_path = ("/").join(new_ds_path.split("/")[:-1])
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

        convlayers = extract_convlayers(
            data_file=params["data_file"],
            h5_dset_path=h5_dset_path,
            model=model,
            seq_length=params["seq_length"],
            time_stride=params["time_stride"],
            lag=params["lag"],
            compute_edge_index=compute_edge_index,
            thres=thres,
        )
        # save the original output to a h5 file
        with h5py.File(output_conv_path, "a") as f:
            for value, key in zip(
                [convlayers],
                ["convlayers"],
            ):
                new_ds_path = h5_dset_path.replace("timeseries", key)
                f[new_ds_path] = value
            # save pneotype info as attributes of the subject
            subject_path = ("/").join(new_ds_path.split("/")[:-1])
            f[subject_path].attrs["site"] = site_dataset_name
            for att in ["sex", "age", "diagnosis"]:
                f[subject_path].attrs[att] = load_data(
                    params["data_file"], h5_dset_path, dtype=att
                )[0]

    df_for_stats.to_csv(output_stats_path, sep="\t")

    print("Plot r2 mean results quickly.")
    # quick plotting
    df_for_stats = df_for_stats[
        (df_for_stats.r2mean > -1e16)
    ]  # remove outliers
    plt.figure()
    g = boxplot(x="site", y="r2mean", hue="diagnosis", data=df_for_stats)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.savefig(output_dir / "figures/diagnosis_by_sites.png")

    plt.figure()
    g = boxplot(x="site", y="r2mean", hue="sex", data=df_for_stats)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.savefig(output_dir / "figures/sex_by_sites.png")


if __name__ == "__main__":
    main()
