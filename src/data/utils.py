from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from joblib import Parallel, delayed
from nilearn.connectome import ConnectivityMeasure
from tqdm import tqdm


def create_connectome(
    data_file: Union[Path, str], dset_paths: List[str]
) -> np.ndarray:
    """Create connectivity matrix with more memory efficient way.

    Args:
      data_file: path to datafile
      dset_path (list of str): path to time series data

    Returns:
      (numpy array): connectivity matrix
    """
    connectome_measure = ConnectivityMeasure(
        kind="correlation", discard_diagonal=True
    )
    avg_corr_mats = None
    for dset in tqdm(
        dset_paths, desc="Computing connectivity matrix group average."
    ):
        data = load_data(
            path=data_file, h5dset_path=dset, standardize=False, dtype="data"
        )
        corr_mat = connectome_measure.fit_transform(data)[0]
        if avg_corr_mats is None:
            avg_corr_mats = corr_mat
        else:
            avg_corr_mats += corr_mat
        del data
        del corr_mat
    avg_corr_mats /= len(dset_paths)
    return avg_corr_mats


def make_sequence(
    data_list: List[np.ndarray],
    length: int,
    stride: int = 1,
    lag: int = 1,
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper to create sequences from a list of time series."""
    if verbose > 2:
        verbose = 50
    X_tot, Y_tot = zip(
        *Parallel(n_jobs=-2, verbose=verbose)(
            delayed(make_sequence_single_subject)(data, length, stride, lag)
            for data in data_list
        )
    )
    return np.concatenate(X_tot), np.concatenate(Y_tot)


def make_sequence_single_subject(
    data: np.ndarray, length: int, stride: int = 1, lag: int = 1
):
    """Slice a list of timeseries with sliding windows and get corresponding labels.

    For each data in data list, pairs generated will correspond to :
    `data[k:k+length]` for the sliding window and `data[k+length+lag-1]` for the label, with k
    iterating with the stride value.

    Args:
        data (numy arrays): data must be of shape (time_steps, features)
        length (int): length of the sliding window
        stride (int): stride of the sliding window (default=1)
        lag (int): time step difference between last time step of sliding window and label time step
            (default=1)
    Returns:
      (tuple): a tuple containing:
        X (numpy array): sliding windows array of shape (nb of sequences, features, length)
        Y (numpy array): labels array of shape (nb of sequences, features)
    """
    X, Y = [], []
    delta = lag - 1
    for i in range(0, data.shape[0] - length - delta, stride):
        X.append(np.moveaxis(data[i : i + length], 0, 1))
        Y.append(data[i + length + delta])
    return np.array(X), np.array(Y)


def load_data(
    path: Union[Path, str],
    h5dset_path: Union[List[str], str],
    standardize: bool = False,
    dtype: str = "data",
) -> List[Union[np.ndarray, str, int, float]]:
    """Load time series or phenotype data from the hdf5 files.

    Args:
        path (Union[Path, str]): Path to the hdf5 file.
        h5dset_path (Union[List[str], str]): Path to data inside the
            h5 file.
        standardize (bool, optional): Whether to standardize the data.
            Defaults to False. Only applicable to dtype='data'.
        dtype (str, optional): Attribute label for each subject or
            "data" to load the time series. Defaults to "data".

    Returns:
        List[Union[np.ndarray, str, int, float]]: loaded data.
    """
    if isinstance(h5dset_path, str):
        h5dset_path = [h5dset_path]
    data_list = []
    if dtype == "data":
        with h5py.File(path, "r") as h5file:
            for p in h5dset_path:
                data_list.append(h5file[p][:])
        if standardize and data_list:
            means = np.concatenate(data_list, axis=0).mean(axis=0)
            stds = np.concatenate(data_list, axis=0).std(axis=0)
            data_list = [(data - means) / stds for data in data_list]
        return data_list
    else:
        with h5py.File(path, "r") as h5file:
            for p in h5dset_path:
                subject_node = "/".join(p.split("/")[:-1])
                data_list.append(h5file[subject_node].attrs[dtype])
        return data_list


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Simple dataset for pytorch training loop"""

    def __init__(self, time_sequence_h5, set_type="train", transform=None):
        self.data = time_sequence_h5
        self.set_type = set_type
        self.transform = transform

    def __len__(self):
        length = self.data[self.set_type]["label"].shape[0]
        return length

    def __getitem__(self, index):
        # read the data
        h5_input_seq = self.data[self.set_type]["input"][index, :, :]
        h5_label = self.data[self.set_type]["label"][index, :]
        # transform to tensors

        if self.transform:
            input_seq = self.transform(h5_input_seq)
            label = self.transform(h5_label)
        else:
            input_seq = torch.from_numpy(h5_input_seq)
            label = torch.from_numpy(h5_label)
        del h5_label
        del h5_input_seq
        return input_seq, label
