from typing import List, Tuple

import h5py
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics import r2_score
from src.data.utils import load_data, make_sequence_single_subject
from src.models.components.chebnet import NonsharedFC
from src.models.components.pooling import StdPool2d
from src.utils import RankedLogger
from torch_geometric.nn import ChebConv

log = RankedLogger(__name__, rank_zero_only=True)


def extract_output_weight(
    model: LightningModule, x: torch.Tensor, edge_index: torch.Tensor
):
    weights = {}
    hooks = []

    def hook_fn(layer, input, output):
        if not weights.get(layer._get_name()):
            weights[layer._get_name()] = []
        # assigning to a variable will create a copy and detach from graph...
        # I hope
        weight = output.detach().cpu().numpy()
        return weights[layer._get_name()].append(weight)

    with torch.no_grad():
        for _, module in model.named_modules():
            if isinstance(module, ChebConv) or isinstance(module, NonsharedFC):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        z = model(x, edge_index).detach().cpu().numpy()

        for hook in hooks:
            hook.remove()
        z = np.expand_dims(z, axis=-1)
    return z, weights


def generate_pooling_funcs(n_parcel, kernel_nodes, kernel_time=1):
    pooling_funcs = {
        # '1dconv': lambda x: nn.Conv2d(in_channels=n_parcel, out_channels=n_parcel, stride=1, kernel_size=(n_output_nodes, kernel_time))(x).detach().numpy().squeeze(),
        "max": lambda x: nn.MaxPool2d(
            kernel_size=(kernel_nodes, kernel_time), stride=1
        )(x)
        .detach()
        .numpy()
        .squeeze(),
        "average": lambda x: nn.AvgPool2d(
            kernel_size=(kernel_nodes, kernel_time), stride=1
        )(x)
        .detach()
        .numpy()
        .squeeze(),
        "std": lambda x: StdPool2d(
            kernel_size=(kernel_nodes, kernel_time), stride=1
        )(x)
        .detach()
        .numpy()
        .squeeze(),
    }
    return pooling_funcs


def predict_horizon(
    horizon: int,
    model: torch.nn.Module,
    data: np.ndarray,
    edge_index: torch.Tensor,
    timeseries_window_stride_lag: Tuple[int, int, int],
    timeseries_decimate: int = 4,
    f: h5py.File = None,
    dset_path: str = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    if f is None != dset_path is None:
        log.error(
            "If you want to save the weights, please pass a file "
            "handler (f) and the h5 dataset path."
        )
        raise ValueError

    w, s, _ = timeseries_window_stride_lag
    m = timeseries_decimate  # decimate factor
    history, _ = make_sequence_single_subject(data, m, w + horizon, s, 0)
    y = history[:, :, w:]
    x = history[:, :, :w]
    del history
    z, layers = [], []
    for h in range(horizon):
        if h > 0:
            # see recursive forecasting:
            # https://www.ethanrosenthal.com/2019/02/18/time-series-for-scikit-learn-people-part3/
            # this is the simplest form of AR forecasting
            # move window forward by one, append predicted time point
            # this is the new input to predict t+h
            x = np.concatenate(
                (x.detach().cpu().numpy()[:, :, 1:], z[-1]), axis=-1
            )
        x = torch.tensor(x, dtype=torch.float32).to(model.device)
        cur_z, cur_weights = extract_output_weight(model, x, edge_index)
        # save prediction results
        z.append(cur_z)
        del cur_z
        # save the all layer weights; size of each layer: (# batches, # ROI, # outputs nodes)
        # 1 batche would be 1 time window here
        layers.append(cur_weights)
        del cur_weights
    z = np.concatenate(z, axis=-1)
    # each item under the key (layer name) in the layer_features dictionary:
    # (# batches, # ROI, # outputs nodes, # horizon)
    restructure = {}
    for horizon_weights in layers:
        for key in horizon_weights:
            if key not in restructure:
                restructure[key] = []
            for i, layer in enumerate(horizon_weights[key]):
                if len(restructure[key]) != len(horizon_weights[key]):
                    restructure[key].append([np.expand_dims(layer, -1)])
                else:
                    restructure[key][i].append(np.expand_dims(layer, -1))
    del layers

    layer_features = {}
    for key in restructure:
        for i, layer in enumerate(restructure[key]):
            layer_features[f"{key}{i+1}"] = np.concatenate(layer, -1)
    del restructure

    # basic validation
    ts_r2 = []
    for h in range(horizon):
        # calculate on ts r2
        ts_r2.append(
            r2_score((y[:, :, h]), z[:, :, h], multioutput="raw_values")
        )
    ts_r2 = np.swapaxes(np.array(ts_r2), 0, -1)
    if f is None and dset_path is None:
        return (y, z, layer_features, ts_r2)

    # save stuff
    conn = ConnectivityMeasure(kind="correlation")
    save_extracted_feature(y, z, layer_features, ts_r2, f, dset_path, conn)
    return None


def save_extracted_feature(
    y: np.ndarray,
    z: np.ndarray,
    layer_features: List[np.ndarray],
    ts_r2: np.ndarray,
    f: h5py.File,
    dset_path: str,
    conn: ConnectivityMeasure,
) -> None:
    horizon = y.shape[-1]

    # need to improve all the following process through chunks
    for value, key in zip([ts_r2, z, y], ["r2map", "Z", "Y"]):
        new_ds_path = dset_path.replace("timeseries", key)
        f[new_ds_path] = value

    # create connectome from real data
    y_conn = conn.fit_transform([y[:, :, 0]])[0]
    new_ds_path = dset_path.replace("timeseries", "connectome")
    f[new_ds_path] = y_conn
    del y_conn
    del y
    # create connectome from simulated data
    z_conns = conn.fit_transform([z[:, :, h] for h in range(horizon)])
    new_ds_path = dset_path.replace("timeseries", "horizon-all_connectome")
    f[new_ds_path] = np.moveaxis(np.array(z_conns), 0, -1)
    del z_conns
    del conn

    # pooling over output of conv layers
    for key, layer in layer_features.items():
        n_time_points, n_parcel, n_output_nodes, _ = layer.shape
        pooling_funcs = generate_pooling_funcs(
            n_parcel, kernel_nodes=1, kernel_time=n_time_points
        )
        new_ds_path = dset_path.replace("timeseries", f"layer-{key}_weights")
        f[new_ds_path] = layer  # save all layers

        # pool over output nodes
        layer_pooled_ts = {method_name: [] for method_name in pooling_funcs}
        for h in range(horizon):
            layer_h = np.moveaxis(
                layer[:, :, :, h], 0, -1
            )  # (# ROI, # outputs nodes, # time windows)
            for method_name in pooling_funcs:
                d = pooling_funcs[method_name](
                    torch.tensor(layer_h).unsqueeze(0)
                )
                layer_pooled_ts[method_name].append(d)

        for method_name in pooling_funcs:
            # save
            new_ds_path = dset_path.replace(
                "timeseries",
                f"layer-{key}_pooling-{method_name}_weights",
            )
            # concatenate along the horizon
            layer_pooled_ts[method_name] = np.moveaxis(
                np.concatenate(
                    np.expand_dims(layer_pooled_ts[method_name], -1), axis=-1
                ),
                0,
                1,
            )
            f[new_ds_path] = layer_pooled_ts[method_name]
        del layer_pooled_ts
