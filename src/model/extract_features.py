import numpy as np
import torch
from fmri_autoreg.data.load_data import make_input_labels
from src.data.load_data import load_data
from torch_geometric.nn import ChebConv


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        layer_activations = module_out
        self.outputs.append(layer_activations.detach().clone())

    def clear(self):
        self.outputs = []


def _module_output_to_numpy(tensor):
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
    """Extract the last conv layer from the pretrained model."""
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
    conv_layers = _module_output_to_numpy(save_output.outputs[-1])
    # get last layers (batch, node, feature F)
    # first layer is nodes, since the rest will be compressed
    # (node, batch, feature F)
    conv_layers = np.swapaxes(conv_layers, 0, 1)
    # remove the hooks
    for handle in hook_handles:
        handle.remove()
    return conv_layers
