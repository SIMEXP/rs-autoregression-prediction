import argparse
import copy
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from giga_companion.load_data import load_data, load_h5_data_path
from src.data.load_data import Dataset, load_params, make_input_labels
from src.models.predict_model import predict_horizon
from src.tools import check_path, load_model, string_to_list
from torch.utils.data import DataLoader
from torch_geometric.nn import ChebConv
from tqdm.auto import tqdm

NUM_WORKERS = 2
BATCH_SIZE = 15


class BinaryPrediction(nn.Module):
    def __init__(self, dropout, n_parcels, FK, bn_momentum=0.1):
        super().__init__()
        FK = string_to_list(FK)
        F = FK[::2]
        n_layers = len(F)
        layers = [
            nn.Linear(
                n_parcels * n_layers * sum(F) * batch_size,
                n_parcels * batch_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(n_parcels * batch_size, momentum=bn_momentum),
            nn.MaxPool3d(n_parcels * batch_size),
            nn.Dropout(dropout),
            nn.Linear(n_parcels * batch_size, 1),  # output
            nn.Sigmoid(),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return self.layers(x).view(x.shape[0], x.shape[1])

    def predict(self, x):
        x = torch.tensor(
            x, dtype=torch.float32, device=next(self.parameters()).device
        )
        self.eval()
        return self.forward(x).cpu().detach().numpy()


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
    # output_dir = Path(output_dir)
    # output_dir.mkdir()

    # load model
    # model_path = (
    #     args.model if args.model.exists() \
    #     else args.model / "model.pkl"
    # )
    model_path = "../outputs/prototype_train_and_test_within-sites_original-tr/model.pkl"

    # get the latent feature per subject

    # load data
    datasets = {}
    for abide in ["abide1", "abide2"]:
        data_dset = load_h5_data_path(
            "../" + params["data_file"],
            f"{abide}.*/*/sub-.*desc-197.*timeseries",
            shuffle=True,
        )
        data_list = load_data(
            "../" + params["data_file"], data_dset, dtype="data"
        )
        labels = load_data(
            "../" + params["data_file"], data_dset, dtype="diagnosis"
        )

        X_lat_feat = []
        for ts in tqdm(data_list):
            X = make_input_labels(
                [ts],
                [],
                params["seq_length"],
                params["time_stride"],
                params["lag"],
                compute_edge_index,
                thres,
            )[
                0
            ]  # just one subject and the X

            # register the hooks
            model = load_model(model_path)
            if isinstance(model, torch.nn.Module):
                model.to(torch.device(device)).eval()

            save_output = SaveOutput()
            hook_handles = []
            for _, module in model.named_modules():
                if isinstance(module, ChebConv):
                    handle = module.register_forward_hook(save_output)
                    hook_handles.append(handle)

            # pass the data through pretrained model
            out = model(torch.tensor(X))
            # conv_layers = np.array(
            #     [
            #         module_output_to_numpy(o)
            #         for o in save_output.outputs
            #     ]
            # )
            conv_layers = module_output_to_numpy(
                save_output.outputs[-1]
            )  # take the last layer
            X_lat_feat.append(conv_layers)
        datasets[abide] = Dataset(X=X_lat_feat, Y=labels)

    # format latent features and prediction labels
    tng_dataloader = DataLoader(
        datasets["abide1"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        datasets["abide2"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    n_epochs = 20  # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_lat_feat), batch_size)

    # Hold the best model
    best_acc = -np.inf  # init to negative infinity
    best_weights = None

    model = BinaryPrediction(
        dropout=0,
        n_parcels=197,
        FK=params["FK"],
    )
    model.to(device)

    for _ in tqdm(range(n_epochs)):
        model.train()
        mean_loss_tng = 0.0
        all_preds_tng = []
        all_labels_tng = []
        all_preds_val = []
        all_labels_val = []
        for sampled_batch in tng_dataloader:
            # take a batch
            inputs = sampled_batch["input"].to(device)
            labels = sampled_batch["label"].to(device)
            # forward pass
            preds = model(inputs)
            loss = loss_fn(preds, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            acc = (preds.round() == labels).float().mean()
            print(f"Training: loss = {loss}; acc = {acc}")

        # evaluate accuracy on validation
        model.eval()
        mean_loss_val = 0.0

        inputs = val_dataloader.inputs.to(device)
        labels = val_dataloader.labels.to(device)
        preds = model(inputs)
        acc = (preds.round() == labels).float().mean()
        acc = float(acc)
        print(f"Validation: acc = {acc}")
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
