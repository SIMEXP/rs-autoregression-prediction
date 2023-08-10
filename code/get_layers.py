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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from giga_companion.load_data import load_data, load_h5_data_path
from src.data.load_data import load_params, make_input_labels
from src.models.predict_model import predict_horizon
from src.tools import check_path, load_model
from torch_geometric.nn import ChebConv
from tqdm import tqdm


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        layer_activations = module_out
        self.outputs.append(layer_activations.detach().clone())

    def clear(self):
        self.outputs = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=Path, help="model output directory"
    )
    parser.add_argument(
        "--param",
        "-p",
        type=Path,
        help="Parameters : path to json file or dict",
    )

    # params = load_params(args.param)
    params = load_params("parameters/prototype.json")
    compute_edge_index = params["model"] == "Chebnet"
    thres = params["edge_index_thres"] if compute_edge_index else None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Working on {device}.")
    # output_dir = args.output_dir
    # output_dir = check_path(output_dir)
    # os.makedirs(output_dir)
    # output_dir = Path(output_dir)

    # load data
    data_dset = load_h5_data_path(
        "../" + params["data_file"],
        "abide2.*/*/sub-.*desc-197.*timeseries",
        shuffle=True,
    )
    data_list = load_data("../" + params["data_file"], data_dset, dtype="data")
    X = make_input_labels(
        [data_list[0]],
        [],
        params["seq_length"],
        params["time_stride"],
        params["lag"],
        compute_edge_index,
        thres,
    )[
        0
    ]  # just one subject and the X

    # load model
    # model_path = (
    #     args.model if args.model.exists() \
    #     else args.model / "model.pkl"
    # )
    model_path = "../outputs/prototype_train_and_test_within-sites_original-tr/model.pkl"
    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()

    # register the hooks
    save_output = SaveOutput()
    hook_handles = []
    for _, module in model.named_modules():
        if isinstance(module, ChebConv):
            handle = module.register_forward_hook(save_output)
            hook_handles.append(handle)

    # pass the data through pretrained model
    out = model(torch.tensor(X))

    print(len(save_output.outputs))

    def module_output_to_numpy(tensor):
        return tensor.detach().to("cpu").numpy()

    conv_layers = module_output_to_numpy(save_output.outputs[0])
