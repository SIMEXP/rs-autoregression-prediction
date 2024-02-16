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


def _module_output_to_numpy(tensor):
    return tensor.detach().to("cpu").numpy()


def extract_convlayers(
    data_file: Union[Path, str],
    h5_dset_path: str,
    model: torch.nn.Module,
    seq_length: int,
    time_stride: int,
    lag: int,
    compute_edge_index: bool,
    thres: float = 0.9,
):
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
    for layer in save_output.outputs:
        convlayers.append(_module_output_to_numpy(layer))
    # remove the hooks
    for handle in hook_handles:
        handle.remove()
    return convlayers


def pooling_convlayers(
    convlayers: List[np.array],
    layer_index: int = -1,
    pooling_methods: str = "average",
):
    """Pooling the conv layers."""
    if pooling_methods == "average":
        m = nn.AdaptiveAvgPool2d((1, 1))
    elif pooling_methods == "max":
        m = nn.AdaptiveMaxPool2d((1, 1))
    elif pooling_methods == "std":
        m = lambda x: torch.std_mean(x, (1, 2))[0]  # (std, mean)
    else:
        raise ValueError(f"Pooling method {pooling_methods} is not supported.")

    data = [torch.tensor(cl, dtype=torch.float32) for cl in convlayers]
    data = torch.cat(data)
    d = m(data).flatten().numpy()
    return d
