"""
Resource:
salloc --time=2:00:00 --mem=4G --cpus-per-task=8 --gpus-per-node=1
>> 16384

Aim:
 - Fill up the GPU memory with the largest batch size possible
"""

import typing as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fmri_autoreg.models.models import Chebnet

DATASET_SIZE = 2328583  # number of data point in training set
SEQ = 16


# make a random correlation matrix
def get_edges(n_emb):
    ts = np.random.rand(n_emb, 117)
    corr = np.corrcoef(ts)
    thres_index = int(corr.shape[0] * corr.shape[1] * 0.9)
    thres_value = np.sort(corr.flatten())[thres_index]
    adj_mat = corr * (corr >= thres_value)
    edge_index = np.nonzero(adj_mat)
    return edge_index


def get_batch_size(
    model: nn.Module,
    device: torch.device,
    input_shape: t.Tuple[int, int, int],
    output_shape: t.Tuple[int],
    dataset_size: int,
    max_batch_size: int = None,
    num_iterations: int = 5,
) -> int:
    model.to(device)
    model.train(True)
    optimizer = torch.optim.Adam(model.parameters())

    batch_size = 2
    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break
        if batch_size >= dataset_size:
            batch_size = batch_size // 2
            break
        try:
            for _ in range(num_iterations):
                # dummy inputs and targets
                inputs = torch.rand(*(batch_size, *input_shape), device=device)
                targets = torch.rand(
                    *(batch_size, *output_shape), device=device
                )
                outputs = model(inputs)
                loss = F.mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            batch_size *= 2
        except RuntimeError:
            batch_size //= 2
            break
    del model, optimizer
    torch.cuda.empty_cache()
    return batch_size


if __name__ == "__main__":
    for n_emb in [64, 197, 444]:
        print("3 conv")
        edge_index = get_edges(n_emb)
        model = Chebnet(
            n_emb=n_emb,
            seq_len=16,
            edge_index=edge_index,
            FK="8,3,8,3,8,3",
            M="8,1",
            FC_type="nonshared_uni",
            aggrs="add",
            dropout=0.1,
            bn_momentum=0.1,
            use_bn=True,
        )
        batch_size = get_batch_size(
            model, torch.device("cuda"), (n_emb, SEQ), (n_emb,), DATASET_SIZE
        )
        print(f"atlas {n_emb}, input length {SEQ}, batch size {batch_size}")
        del model
        print("6 conv")
        model = Chebnet(
            n_emb=n_emb,
            seq_len=16,
            edge_index=edge_index,
            FK="8,3,8,3,8,3,8,3,8,3,8,3",
            M="8,1",
            FC_type="nonshared_uni",
            aggrs="add",
            dropout=0.1,
            bn_momentum=0.1,
            use_bn=True,
        )
        batch_size = get_batch_size(
            model, torch.device("cuda"), (n_emb, SEQ), (n_emb,), DATASET_SIZE
        )
        print(f"atlas {n_emb}, input length {SEQ}, batch size {batch_size}")
        del model
        del edge_index
