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
MAX_TIME_SEQ = 100
LEARNING_RATE = 0.1
LEARNING_RATE_PATIENCE = 4
LEARNING_RATE_THRES = 0.01
WEIGHT_DECAY = 0
DROP_OUT = 0.2


def zero_padding(conv_layers, maxium_time_sequence):
    if conv_layers.shape[0] < maxium_time_sequence:
        conv_layers = np.pad(
            conv_layers,
            ((0, maxium_time_sequence - conv_layers.shape[0]), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    else:
        conv_layers = conv_layers[:maxium_time_sequence, ::]
    return conv_layers


class BinaryPrediction(nn.Module):
    def __init__(
        self, dropout, n_parcels, FK, batch_size, n_time_seq, bn_momentum=0.1
    ):
        super().__init__()
        FK = string_to_list(FK)
        F = FK[::2]
        # n_layers = len(F)
        n_latent_features = F[-1]  # using the last layer only for now
        fc_out = 128
        kernel_size = 3
        stride = 2
        max_pool_output = np.ceil((fc_out - kernel_size - 1 - 1) / stride + 1)
        max_pool_output = int(max_pool_output)

        layers = [
            nn.Linear(
                n_parcels * n_latent_features,
                fc_out,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(n_time_seq, momentum=bn_momentum),
            nn.MaxPool1d(kernel_size, stride=stride),
            nn.Linear(max_pool_output, 1),  # output
            nn.Sigmoid(),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.view((x.shape[0], x.shape[1]))

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
    datasets = {}
    for abide in ["abide1", "abide2"]:
        # load data
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
        Y = []
        # get the convolution layer features per subject
        for ts, label in tqdm(zip(data_list, labels)):
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

            # do I need to reload the model every time?
            model = load_model(model_path)
            if isinstance(model, torch.nn.Module):
                model.to(torch.device(device)).eval()

            # register the hooks to the pretrain model
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
            # )  # get all layers
            conv_layers = module_output_to_numpy(
                save_output.outputs[-1]
            )  # take the last layer for now
            # use the first 100 time points, pad with zero if not long
            # enough
            conv_layers = zero_padding(conv_layers, MAX_TIME_SEQ)
            # keep the first dimension (time), flatten it
            conv_layers = conv_layers.reshape(conv_layers.shape[0], -1)

            X_lat_feat.append(conv_layers)
            Y.append(np.repeat([label], MAX_TIME_SEQ, axis=0))
        datasets[abide] = Dataset(X=X_lat_feat, Y=Y)

    # format latent features and prediction labels
    tng_dataloader = DataLoader(
        datasets["abide1"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    # just evaluate the full abide 2 at each epoch
    val_inputs = torch.stack(
        [torch.tensor(d) for d in datasets["abide2"].inputs]
    )
    val_labels = torch.stack(
        [torch.tensor(d) for d in datasets["abide2"].labels]
    )
    # val_dataloader = DataLoader(
    #     datasets["abide2"],
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     drop_last=True,
    #     num_workers=NUM_WORKERS,
    # )

    dx_model = BinaryPrediction(
        dropout=DROP_OUT,
        n_parcels=197,
        FK=params["FK"],
        batch_size=BATCH_SIZE,
        n_time_seq=MAX_TIME_SEQ,
    )

    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
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
        for sampled_batch in tng_dataloader:
            # take a batch
            inputs = sampled_batch["input"].to(device)
            labels = sampled_batch["label"].to(device)
            # forward pass
            preds = dx_model(inputs)
            loss = loss_fn(preds, labels)
            mean_loss_tng += loss.item()
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

            # save batch accuracy
            acc = (preds.round() == labels).float().mean()
            mean_acc_tng += acc

        mean_loss_tng = mean_loss_tng / len(tng_dataloader)
        mean_acc_tng = mean_acc_tng / len(tng_dataloader)
        scheduler.step(mean_loss_tng)
        print(
            f"Training: avg loss = {mean_loss_tng}; avg acc = {mean_acc_tng}"
        )

        # evaluate accuracy on validation
        dx_model.eval()
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)
        val_preds = dx_model(val_inputs)
        acc = (val_preds.round() == val_labels).float().mean()
        acc = float(acc)
        print(f"Validation: acc = {acc}")
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(dx_model.state_dict())
    # restore model and return best accuracy
    dx_model.load_state_dict(best_weights)
