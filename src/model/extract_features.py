from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from fmri_autoreg.data.load_data import make_input_labels
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
    compute_edge_index: bool,
    thres: float = 0.9,
) -> List[torch.tensor]:
    """Extract conv layers from the pretrained model for one subject."""
    # load data. No standardisation as it's already done.
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
    device = next(model.parameters()).device
    # pass the data through pretrained model
    _ = model(torch.tensor(X_ts).to(device))
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
    layer_index: int = -1,
    layer_structure: Tuple[int] = None,
) -> np.array:
    """Pooling the conv layers.

    Args:
        convlayers (torch.tensor) : shape
            (time series, parcel, stack layer feature F)
        layer_index (int) : the index of the layer to be pooled, -1
            means pooling all layers.
        pooling_methods (str) : "average", "max", "std"
        pooling_target (str) : "parcel", "timeseries"

    Returns:
        np.array: the pooled feature, shape (parcel, ) if
            pooling_target is "parcel", or shape (time series, parcel)
            if pooling_target is "timeseries"
    """
    if pooling_methods not in ["average", "max", "std"]:
        raise ValueError(f"Pooling method {pooling_methods} is not supported.")
    if pooling_target not in ["parcel", "timeseries"]:
        raise ValueError(f"Pooling target {pooling_target} is not supported.")
    if layer_index > len(layer_structure):
        raise ValueError(
            "The layer index should be smaller than the length of the "
            f"layer structure. layer index is {layer_index} but there "
            f"are {len(layer_structure)} layers."
        )
    if layer_structure is None and layer_index != -1:
        raise ValueError(
            "The layer structure should be provided if layer index is "
            "not -1."
        )

    if layer_index != -1:  # select the layer to be pooled
        if sum(layer_structure) != convlayers.shape[-1]:
            raise ValueError(
                "The sum of layer structure should be equal to the "
                "feature dimension of the convlayers."
            )
        # unstack the convlayers by layer structure
        structure_layers = []
        for i, f in enumerate(layer_structure):
            i = sum(layer_structure[0:i])
            j = i + f
            structure_layers.append(convlayers[:, :, i:j])
        convlayers = structure_layers[layer_index]
        return pooling_convlayers(
            convlayers,
            pooling_methods=pooling_methods,
            pooling_target=pooling_target,
        )

    # pooling the convlayers
    n_parcel = convlayers.shape[1]

    if pooling_target == "parcel":
        convlayers = torch.swapaxes(convlayers, 0, 1)
        # (parcel, time series, stack layer feature F)
        output_size = (1, 2) if pooling_methods == "std" else (1, 1)
    if pooling_target == "timeseries":
        output_size = (2) if pooling_methods == "std" else (n_parcel, 1)

    if pooling_methods == "average":
        m = nn.AdaptiveAvgPool2d(output_size)
    elif pooling_methods == "max":
        m = nn.AdaptiveMaxPool2d(output_size)
    elif pooling_methods == "std":
        m = lambda x: torch.std_mean(x, output_size)[0]  # (std, mean)
    else:
        raise ValueError(f"Pooling method {pooling_methods} is not supported.")

    d = m(convlayers[0]).numpy().squeeze()
    return d
