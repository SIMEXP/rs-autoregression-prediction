from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from fmri_autoreg.data.load_data import make_seq
from src.data.load_data import load_data
from torch_geometric.nn import ChebConv


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        layer_activations = module_out
        self.outputs.append(layer_activations.detach().to("cpu").clone())

    def clear(self):
        self.outputs = []


def _module_output_to_cpu(tensor):
    return tensor.detach().to("cpu")


def extract_convlayers(
    data_file: Union[Path, str],
    h5_dset_path: str,
    model: torch.nn.Module,
    seq_length: int,
    time_stride: int,
    lag: int,
) -> torch.tensor:
    """Extract conv layers from the pretrained model for one subject."""
    with torch.no_grad():
        # load data. No standardisation as it's already done.
        ts = load_data(data_file, h5_dset_path, dtype="data")
        X_ts = make_seq(
            ts,
            seq_length,
            time_stride,
            lag,
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
        device = next(model.parameters()).device
        # pass the data through pretrained model
        X_ts = torch.tensor(X_ts, dtype=torch.float32, device=device)
        _ = model(X_ts)
        convlayers = []
        # size of each layer (time series, parcel, layer feature F)
        for layer in save_output.outputs:
            layer = _module_output_to_cpu(layer)
            convlayers.append(layer)
        # stack along the feature dimension
        convlayers = torch.cat(convlayers, dim=-1)
        # remove the hooks
        for handle in hook_handles:
            handle.remove()
    return convlayers


def pooling_convlayers(
    convlayers: torch.tensor,
    pooling_methods: str = "average",
    pooling_target: str = "parcel",
    layer_index: int = -99,
    layer_structure: Tuple[int] = None,
) -> np.array:
    """Pooling the conv layers.

    Args:
        convlayers (torch.tensor) : shape
            (time series, parcel, stack layer feature F)
        layer_index (int) : the index of the layer to be pooled, -99
            means pooling all layers.
        pooling_methods (str) : "average", "max", "std"
        pooling_target (str) : keep "parcel" or "timeseries" and parcels

    Returns:
        np.array: the pooled feature, shape (parcel, ) if
            pooling_target is "parcel", or shape (time series, parcel)
            if pooling_target is "timeseries"
    """
    if pooling_methods not in ["average", "max", "std", "1dconv"]:
        raise ValueError(f"Pooling method {pooling_methods} is not supported.")
    if pooling_target not in ["parcel", "timeseries"]:
        raise ValueError(f"Pooling target {pooling_target} is not supported.")
    if layer_structure and layer_index > len(layer_structure):
        raise ValueError(
            "The layer index should be smaller than the length of the "
            f"layer structure. layer index is {layer_index} but there "
            f"are {len(layer_structure)} layers."
        )

    if layer_index != -99:  # select the layer to be pooled
        if sum(layer_structure) != convlayers.shape[-1]:
            raise ValueError(
                "The sum of layer structure should be equal to the "
                "feature dimension of the convlayers."
            )
        # unstack the convlayers by layer structure
        start_index = sum(layer_structure[:layer_index])
        end_index = start_index + layer_structure[layer_index]
        convlayers = convlayers[:, :, start_index:end_index]
        return pooling_convlayers(
            convlayers,
            pooling_methods=pooling_methods,
            pooling_target=pooling_target,
        )
    # start with (time series, parcel, stack layer feature F)
    # (parcel, time series, stack layer feature F)
    convlayers = torch.swapaxes(convlayers, 0, 1)
    # (parcel, stack layer feature F, time series)
    convlayers = torch.swapaxes(convlayers, 1, 2)
    n_parcel, n_feature, n_time = convlayers.size()

    if pooling_methods == "1dconv":
        in_channels = n_feature
        out_channels = 1  # convolving over layer features
        kernel_size = n_time if pooling_target == "parcel" else 1
        m = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
    elif pooling_methods in ["average", "max"]:
        # add channel dimension to convlayers
        # (parcel, channel, stack layer feature F, time series)
        # treat each parcel as independent batches
        # channel is always 1
        convlayers = convlayers.unsqueeze(1)
        kernel_size = (
            (n_feature, n_time)
            if pooling_target == "parcel"
            else (n_feature, 1)
        )
        if pooling_methods == "average":
            m = nn.AvgPool2d(kernel_size=kernel_size)
        elif pooling_methods == "max":
            m = nn.MaxPool2d(kernel_size=kernel_size)
    elif pooling_methods == "std":
        # calculate along which dimensions
        dim = (1, 2) if pooling_target == "parcel" else (1)
        m = lambda x: torch.std_mean(x, dim)[0]
    else:
        raise ValueError(f"Pooling method {pooling_methods} is not supported.")
    if pooling_methods != "1dconv":
        d = m(convlayers).numpy().squeeze()
    else:
        d = m(convlayers).detach().numpy().squeeze()
    return d
