from typing import List, Tuple

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics import r2_score
from src.data.utils import load_data, make_sequence_single_subject
from src.models.components.pooling import StdPool2d
from torch_geometric.nn import ChebConv


def extract_chebnet_output_weight(
    model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor
):
    weights = []
    hooks = []

    def hook_fn(layer, input, output):
        return weights.append(output.detach().numpy())

    for _, module in model.named_modules():
        if isinstance(module, ChebConv):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

    with torch.no_grad():
        z = model.predict(x, edge_index)

    for hook in hooks:
        hook.remove()
    z = np.expand_dims(z.detach().numpy(), axis=-1)
    return z, weights


def generate_pooling_funcs(n_parcel, n_output_nodes, kernel_time=1):
    pooling_funcs = {
        # '1dconv': lambda x: nn.Conv2d(in_channels=n_parcel, out_channels=n_parcel, stride=1, kernel_size=(n_output_nodes, kernel_time))(x).detach().numpy().squeeze(),
        "max": lambda x: nn.MaxPool2d(
            kernel_size=(n_output_nodes, kernel_time), stride=1
        )(x)
        .detach()
        .numpy()
        .squeeze(),
        "average": lambda x: nn.AvgPool2d(
            kernel_size=(n_output_nodes, kernel_time), stride=1
        )(x)
        .detach()
        .numpy()
        .squeeze(),
        "std": lambda x: StdPool2d(
            kernel_size=(n_output_nodes, kernel_time), stride=1
        )(x)
        .detach()
        .numpy()
        .squeeze(),
    }
    return pooling_funcs


def predict_horizon(
    horizon: int,
    net: torch.nn.Module,
    data: np.ndarray,
    edge_index: torch.Tensor,
    timeseries_window_stride_lag: Tuple[int, int, int],
    timeseries_decimate: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    w, s, _ = timeseries_window_stride_lag
    m = timeseries_decimate  # decimate factor
    history, _ = make_sequence_single_subject(data, m, w + horizon, s, 0)
    y = history[:, :, w:]
    x = history[:, :, :w]
    z, convlayers = [], []
    for h in range(horizon):
        if h > 0:
            # see recursive forcasting:
            # https://www.ethanrosenthal.com/2019/02/18/time-series-for-scikit-learn-people-part3/
            # this is the simplist form of AR forcasting
            # move window forward by one, append predicted time point
            # this is the new input to predict t+h
            x = np.concatenate((x.detach().numpy()[:, :, 1:], z[-1]), axis=-1)
        x = torch.tensor(x, dtype=torch.float32)
        cur_z, cur_weights = extract_chebnet_output_weight(net, x, edge_index)
        # save prediction results
        z.append(cur_z)
        # save the all layer weights; size of each layer: (# batches, # ROI, # outputs nodes)
        # 1 batche would be 1 time window here
        for i, l in enumerate(cur_weights):
            if len(convlayers) != len(cur_weights):
                convlayers.append([np.expand_dims(l, -1)])
            else:
                convlayers[i].append(np.expand_dims(l, -1))
    z = np.concatenate(z, axis=-1)
    cleaned_conv = (
        []
    )  # each item: (# batches, # ROI, # outputs nodes, # horizon)
    for l in convlayers:
        cleaned_conv.append(np.concatenate(l, -1))

    # basic validation
    ts_r2 = []
    for h in range(horizon):
        # calculate on ts r2
        ts_r2.append(
            r2_score((y[:, :, h]), z[:, :, h], multioutput="raw_values")
        )
    ts_r2 = np.swapaxes(np.array(ts_r2), 0, -1)
    return (y, z, cleaned_conv, ts_r2)


def save_extracted_feature(
    ts: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    convlayers: List[np.ndarray],
    ts_r2: np.ndarray,
    f: h5py.File,
    dset_path: str,
) -> None:
    horizon = y.shape[-1]
    f[dset_path] = ts

    for value, key in zip([ts_r2, z, y], ["r2map", "Z", "Y"]):
        new_ds_path = dset_path.replace("timeseries", key)
        f[new_ds_path] = value

    # create connectome from real data
    conn = ConnectivityMeasure(kind="correlation")
    y_conn = conn.fit_transform([y[:, :, 0]])[0]
    new_ds_path = dset_path.replace("timeseries", "connectome")
    f[new_ds_path] = y_conn

    # create connectome from simulated data
    z_conns = conn.fit_transform([z[:, :, h] for h in range(horizon)])
    new_ds_path = dset_path.replace("timeseries", "horizon-all_connectome")
    f[new_ds_path] = np.moveaxis(np.array(z_conns), 0, -1)

    # pooling over output of conv layers
    _, n_parcel, n_output_nodes, _ = convlayers[0].shape
    pooling_funcs = generate_pooling_funcs(
        n_parcel, n_output_nodes, kernel_time=1
    )
    for i, layer in enumerate(convlayers):
        new_ds_path = dset_path.replace(
            "timeseries", f"layer-{i+1}_gcnweights"
        )
        f[new_ds_path] = layer  # save all layers

        # pool over output nodes
        layer_pooled_ts = {method_name: [] for method_name in pooling_funcs}
        layer_pooled_conn = {method_name: [] for method_name in pooling_funcs}
        for h in range(horizon):
            layer_h = np.moveaxis(
                layer[:, :, :, h], 0, -1
            )  # (# ROI, # outputs nodes, # time windows)
            for method_name in pooling_funcs:
                d = pooling_funcs[method_name](
                    torch.tensor(layer_h).unsqueeze(0)
                )
                layer_pooled_ts[method_name].append(d)
                d_conn = conn.fit_transform([d.T])[0]
                layer_pooled_conn[method_name].append(d_conn)

        for method_name in pooling_funcs:
            # concatenate along the horizon
            layer_pooled_ts[method_name] = np.moveaxis(
                np.concatenate(
                    np.expand_dims(layer_pooled_ts[method_name], -1), axis=-1
                ),
                0,
                1,
            )
            layer_pooled_conn[method_name] = np.concatenate(
                np.expand_dims(layer_pooled_conn[method_name], -1), axis=-1
            )

            # save
            new_ds_path = dset_path.replace(
                "timeseries",
                f"layer-{i+1}_pooling-{method_name}_gcnweights_timeseries",
            )
            f[new_ds_path] = layer_pooled_ts[method_name]

            new_ds_path = dset_path.replace(
                "timeseries",
                f"layer-{i+1}_pooling-{method_name}_gcnweights_connectome",
            )
            f[new_ds_path] = layer_pooled_conn[method_name]
