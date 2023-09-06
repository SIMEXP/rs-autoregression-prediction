import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from giga_companion.load_data import load_data, load_h5_data_path
from src.data.load_data import load_params, make_input_labels
from src.tools import check_path, load_model
from torch_geometric.nn import ChebConv
from tqdm.auto import tqdm

MAX_TIME_SEQ = 100


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
        "--output_dir", "-o", type=Path, help="output directory"
    )
    parser.add_argument(
        "--model", "-m", type=Path, help="model output directory"
    )
    parser.add_argument(
        "--param",
        "-p",
        type=Path,
        help="Parameters : path to json file or dict",
    )
    args = parser.parse_args()

    params = load_params(args.param)
    compute_edge_index = params["model"] == "Chebnet"
    thres = params["edge_index_thres"] if compute_edge_index else None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Working on {device}.")

    output_dir = args.output_dir
    output_dir = check_path(output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir()
    output_metric_path = output_dir / "convlayers.h5"

    # load model
    model_path = (
        args.model if args.model.exists() else args.model / "model.pkl"
    )
    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()

    # get the latent feature per subject
    datasets = {}
    for abide in ["abide1", "abide2"]:
        # load data
        data_dset = load_h5_data_path(
            params["data_file"],
            f"{abide}.*/*/sub-.*desc-197.*timeseries",
            shuffle=True,
        )

        X_lat_feat = []
        Y = []
        # get the convolution layer features per subject
        for h5_dset_path in tqdm(data_dset):
            site_dataset_name = h5_dset_path.split("/")[1]
            h5_dset_name = h5_dset_path.split("/")[-1]
            ts = load_data(params["data_file"], h5_dset_path, dtype="data")
            dx = load_data(
                params["data_file"], h5_dset_path, dtype="diagnosis"
            )
            X = make_input_labels(
                ts,
                [],
                params["seq_length"],
                params["time_stride"],
                params["lag"],
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
            out = model(torch.tensor(X))
            conv_layers = np.array(
                [module_output_to_numpy(o) for o in save_output.outputs]
            )  # get all layers (layer, batch, node, feature F)
            # let first layer be batch / time dimension
            # (batch, layer, node, feature F)
            conv_layers = np.swapaxes(conv_layers, 0, 1)

            # remove the hooks
            for handle in hook_handles:
                handle.remove()

            # save conv_layers to hdf5
            # save the original output to a h5 file
            with h5py.File(output_metric_path, "a") as f:
                # save pneotype info as attributes of the subject
                conv_path = h5_dset_path.replace("timeseries", "convlayers")
                f[conv_path] = conv_layers
                subject_path = ("/").join(h5_dset_path.split("/")[:-1])
                f[subject_path].attrs["site"] = site_dataset_name
                for att in ["sex", "age", "diagnosis"]:
                    f[subject_path].attrs[att] = load_data(
                        params["data_file"], h5_dset_path, dtype=att
                    )[0]
