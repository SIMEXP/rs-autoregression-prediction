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

N_EPOCHS = 40
NUM_WORKERS = 2
BATCH_SIZE = 15
LEARNING_RATE = 0.1
LEARNING_RATE_PATIENCE = 4
LEARNING_RATE_THRES = 0.01
WEIGHT_DECAY = 0
DROP_OUT = 0.2
device = "cpu"


class BinaryPrediction(nn.Module):
    def __init__(self, n_in, n_hidden="256,64", dropout=0.2, bn_momentum=0.1):
        super().__init__()
        n_hidden = string_to_list(n_hidden)
        layers = []
        for i, n in enumerate(n_hidden):
            if i == 0:
                layers.append(nn.Linear(n_in, n))
            else:
                layers.append(nn.Linear(n_hidden[i - 1], n))

            if i < len(n_hidden):
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout())
                layers.append(nn.BatchNorm1d(n))
        layers.append(nn.Linear(n_hidden[-1], 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.view((x.shape[0], x.shape[1])).squeeze()

    def predict(self, x):
        x = torch.tensor(
            x, dtype=torch.float32, device=next(self.parameters()).device
        )
        self.eval()
        return self.forward(x).cpu().detach().numpy()


if __name__ == "__main__":
    # load conv layers
    path_conv_layers = Path(
        "outputs/prototype_train_and_test_within-sites_original-tr/convlayers.h5"
    )
    dataloaders = {}

    m = nn.AdaptiveMaxPool3d((10, 3, 8))
    # m = nn.AdaptiveAvgPool3d((1, 3, 8))

    for abide in ["abide1", "abide2"]:
        dset_path = load_h5_data_path(
            path_conv_layers,
            f"{abide}.*/*/sub-.*desc-197.*",
            shuffle=True,
        )
        conv_layers = []
        for d in dset_path:
            d = load_data(path_conv_layers, d, dtype="data")[0]
            d = torch.tensor(d, dtype=torch.float32)
            d = m(d).flatten()
            conv_layers.append(d)
        dx = [
            load_data(path_conv_layers, d, dtype="diagnosis")[0]
            for d in dset_path
        ]

        # format latent features and prediction labels
        dataloaders[abide] = DataLoader(
            Dataset(X=conv_layers, Y=dx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=NUM_WORKERS,
        )
        del conv_layers, dx
    print(d.shape)
    dx_model = BinaryPrediction(
        dropout=DROP_OUT,
        n_in=197 * 10 * 3 * 8,
        n_hidden="128,64,64,64",
    )

    # loss function and optimizer
    loss_function = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(dx_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=LEARNING_RATE_PATIENCE,
        threshold=LEARNING_RATE_THRES,
    )

    # Hold the best model
    best_acc = -np.inf  # init to negative infinity
    best_weights = None

    dx_model.to(device)

    for _ in tqdm(range(N_EPOCHS)):
        dx_model.train()
        mean_loss_tng = 0.0
        mean_acc_tng = 0.0
        for sampled_batch in dataloaders["abide1"]:
            # take a batch
            inputs = sampled_batch["input"].to(device)
            labels = sampled_batch["label"].to(device)
            # forward pass
            preds = dx_model(inputs)
            loss = loss_function(preds, labels)
            mean_loss_tng += loss.item()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

            # save batch accuracy
            acc = (preds.round() == labels).float().mean()
            mean_acc_tng += acc

        mean_loss_tng = mean_loss_tng / len(dataloaders["abide1"])
        mean_acc_tng = mean_acc_tng / len(dataloaders["abide1"])
        scheduler.step(mean_loss_tng)
        print(
            f"Training: avg loss = {mean_loss_tng}; avg acc = {mean_acc_tng}"
        )

        # evaluate accuracy on validation
        dx_model.eval()
        mean_loss_val = 0.0
        mean_acc_val = 0.0
        for sampled_batch in dataloaders["abide2"]:
            # take a batch
            inputs = sampled_batch["input"].to(device)
            labels = sampled_batch["label"].to(device)
            preds = dx_model(inputs)
            loss = loss_function(preds, labels)
            mean_loss_val += loss.item()
            acc = (preds.round() == labels).float().mean()
            acc = float(acc)
            mean_acc_val += acc
        mean_loss_val = mean_loss_val / len(dataloaders["abide2"])
        mean_acc_val = mean_acc_val / len(dataloaders["abide2"])
        print(
            f"Validation: avg loss = {mean_loss_val}; avg acc = {mean_acc_val}"
        )
