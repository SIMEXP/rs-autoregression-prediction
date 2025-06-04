import itertools
from pathlib import Path

import pandas as pd
import torch
from calflops import calculate_flops
from src.models.autoreg_module import GraphAutoRegModule
from src.models.components.chebnet import Chebnet, get_edge_index_threshold
from tqdm import tqdm

time_sequence_file = "inputs/data/atlas-MIST197_decimate-4_windowsize-16_stride-1_lag-1_seed-1_data.h5"
batch_size = 128
window_size = 16
n_parcel = 197
GCL_options = [3, 6, 9, 12, 24]
F_options = [8, 16, 32, 64]
K_options = [3, 5, 10]
FCL_options = [1, 3, 5, 10]
M_options = [8, 16, 32]


def get_architecture(GCL, F, K, FCL, M):
    FK = str(f"{F},{K}," * GCL)[:-1]
    M = f"{M}," * FCL + "1"
    return FK, M


def gcn_flops(GCL, F, K, FCL, M):
    edge = torch.tensor(
        get_edge_index_threshold(
            time_sequence_file=time_sequence_file, connectome_threshold=0.9
        )
    )
    FK, M_ = get_architecture(GCL, F, K, FCL, M)

    net = Chebnet(
        n_emb=n_parcel,
        seq_len=16,
        FK=FK,
        M=M_,
        FC_type="nonshared_uni",
        aggrs="add",
        dropout=0,
        bn_momentum=0.1,
    )
    optimizer = torch.optim.Adam(
        params=net.parameters(), lr=0.01, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.1, patience=4
    )
    model = GraphAutoRegModule(
        n_regions=n_parcel,
        edge_index=edge,
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        compile=False,
    )
    input_shape = (
        batch_size,
        n_parcel,
        window_size,
    )
    x = torch.ones(()).new_empty(
        (*input_shape,), dtype=next(model.parameters()).dtype
    )
    flops, macs, params = calculate_flops(
        model=model,
        output_as_string=False,
        print_results=False,
        output_unit=None,
        kwargs={"x": x, "edge_index": edge},
    )
    return flops, macs, params


all_combbinations = list(
    itertools.product(
        *[GCL_options, F_options, K_options, FCL_options, M_options]
    )
)

calculated_flops = []
flag = "w"
for current_set in tqdm(all_combbinations):
    GCL, F, K, FCL, M = current_set
    name = (
        f"N-{n_parcel}_W-{window_size}_GCL-{GCL}_F-{F}_K-{K}_FCL-{FCL}_M-{M}"
    )
    if Path("outputs/performance_info/calculated_flops.tsv").is_file():
        with open("outputs/performance_info/calculated_flops.tsv") as f:
            if name in f.read():
                continue
        flag = "a"

    flops, macs, params = gcn_flops(GCL, F, K, FCL, M)
    with open("outputs/performance_info/calculated_flops.tsv", flag) as f:
        if flag == "w":
            f.write("name\tflops\tmacs\tparams\n")
        f.write(f"{name}\t{flops}\t{macs}\t{params}\n")
