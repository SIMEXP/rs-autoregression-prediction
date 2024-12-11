from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv


class Chebnet(nn.Module):
    """Chebnet model.
    Pytorch adaptation of the model proposed in https://github.com/mdeff/cnn_graph

    Args:
        n_emb (int): number of nodes
        seq_len (int): sequence length
        edge_index (tuple of numpy arrays): edges of the graph connectivity in Coordinate Format (COO)
        FK (str): string of comma-separated integers, interlaced list of F and K with:
            F the list of the numbers of output features for each layer (e.g. '4,4,2')
            K the list of the orders of the chebychev polynomial for each layer
            The list is a str with comma separated digits, e.g. for F = [16,8,8] and K = [3,3,1] FK is
            '16,3,8,3,8,1'.
        M (str): dimensionality of FC layers, string of comma-separated integers
        FC_type (str): type of FC layers, choices are :
            'shared_uni': every node use the same FC layer that uses as input only the features of
                the node (univariate)
            'nonshared_uni': each node uses a specific FC layer that uses as input only the features
                of the node (univariate)
            'multi': output of each node is computed from features of every node (multivariate)
        aggrs (str): aggregation methods for ChebConv layers, string of comma-seperated strings will
            be used for each layer. If no comma is presented, all layers will use the same
            aggrefation method (default='add').
        dropout (float): probability of an element being zeroed (default=0)
        bn_momentum (float): momentum for batch normalisation (default=0.1)
        **chevconv_kwargs (optional): additional arguments for ChebConv
    """

    def __init__(
        self,
        n_emb: int,
        seq_len: int,
        FK: str,
        M: str,
        FC_type: str,
        aggrs: str = "add",
        dropout: float = 0,
        bn_momentum: float = 0.1,
    ) -> None:
        super(Chebnet, self).__init__()

        self.FC_type = FC_type
        FK = string_to_list(FK)
        F = FK[::2]
        K = FK[1::2]
        M = string_to_list(M)
        F.insert(0, seq_len)
        aggrs = aggrs.split(",") if "," in aggrs else [aggrs] * len(K)

        if FC_type == "shared_uni":
            make_FC_layer = nn.Linear
        elif FC_type == "nonshared_uni":
            make_FC_layer = lambda f_in, f_out: NonsharedFC(n_emb, f_in, f_out)
        elif FC_type == "multi":
            make_FC_layer = lambda f_in, f_out: MultiFC(n_emb, f_in, f_out)
        else:
            raise ValueError(
                f"Invalid FC_type : '{FC_type}'. Valid values are 'shared_uni', 'nonshared_uni', 'multi'."
            )

        layers = []
        for i in range(len(K)):
            layers.append(
                ChebConv(
                    in_channels=F[
                        i
                    ],  # size of each input sample in time dimension
                    out_channels=F[i + 1],
                    K=K[i],  # number of filters
                    normalization="sym",  # chev conv default
                    bias=True,  # chev conv default
                    aggr=aggrs[i],
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(n_emb, momentum=bn_momentum))
            layers.append(nn.Dropout(dropout))

        for i in range(len(M)):
            layers.append(make_FC_layer(M[i - 1], M[i]))
            if i < len(M) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(n_emb, momentum=bn_momentum))
                layers.append(nn.Dropout(dropout))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            if "ChebConv" in layer.__str__():
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x.view((x.shape[0], x.shape[1]))


class MultiFC(nn.Module):
    """Fully connected layer, connecting all nodes together (for multivariate models).

    Args:
        n (int): number of nodes
        f_in (int): dimension of input signal of each node
        f_out (int): dimension of output signal of each node
    """

    def __init__(self, n, f_in, f_out):
        super(MultiFC, self).__init__()
        self.fc = nn.Linear(n * f_in, n * f_out)
        self.n = n
        self.f_out = f_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.view((x.shape[0], -1))
        y = self.fc(y)
        y = y.view((x.shape[0], self.n, self.f_out))
        return y


class NonsharedFC(nn.Module):
    """Layer connecing independently features of each node, with different weights for each node
    (for univariate models).

    Args:
        n (int): number of nodes
        f_in (int): dimension of input signal of each node
        f_out (int): dimension of output signal of each node
        init_scale (float): scale of the normal initialization of weights (default=0.1)
    """

    def __init__(self, n, f_in, f_out, init_scale=0.1):
        super(NonsharedFC, self).__init__()
        self.w = nn.Parameter(init_scale * torch.randn(n, f_in, f_out))
        self.b = nn.Parameter(init_scale * torch.randn(n, f_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, :, :, None] * self.w[None, :, :, :]).sum(2) + self.b


def get_edge_index_threshold(
    time_sequence_file: str, connectome_threshold: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the edge index of the graph from thresholded functional connectome.

    Args:
        time_sequence_file (str): path to the data h5 file
        connectome_threshold (float): threshold value for the connectome (default=0.9)

    Returns:
        edge_index (tuple of np.ndarrays):
            edge index of the graph in Coordinate Format (COO)
    """
    time_sequence_h5 = h5py.File(time_sequence_file, "r")
    connectome = time_sequence_h5["connectome"][:]
    thres_index = int(
        connectome.shape[0] * connectome.shape[1] * connectome_threshold
    )
    thres_value = np.sort(connectome.flatten())[thres_index]
    adj_mat = connectome * (connectome >= thres_value)
    del connectome
    del thres_value
    edge_index = np.nonzero(adj_mat)
    del adj_mat
    return np.array(edge_index)


def string_to_list(L: str) -> List[int]:
    """Turn a string of comma separated digits to a list of int."""
    return [int(el) for el in L.split(",")]
