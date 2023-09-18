import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
from sklearn.model_selection import train_test_split


def split_data_by_site(
    path: Path,
    datasets: List[str],
    test_set: Union[float, None] = None,
    split_type: Union[str, None] = None,
    data_filter: Union[str, None] = None,
) -> Tuple[List[np.ndarray]]:
    """Train-test split with different strategies for multiple datasets.

    Args:
        path (Path): Path to the h5 file with all the data.
        datasets (List[str]): a list of dataset names to process.
        test_set (Union[float, None]): proportion of the test set size.
            If None, use scikit learn default (0.25).
        split_type (Union[str, None])): support `between_site`,
            `within_site`. If None, and None for data_filter,
            shuffle everything.
        data_filter (Union[str, None])): Regex pattern to look for
            suitable data, such as the right `desc` field for atlas,
            e.g., "*/*/sub-.*desc-197.*timeseries".

    Returns:
        Tuple[List[np.ndarray]]: training data and testing data.
    """
    if len(data_filter.split("/")) != 3:
        raise ValueError(
            "The `data_filter` is not the correct format. It should be "
            "a regex pattern of with three parts, based on the h5 file,"
            " i.e. '<dataset>/<subject>/<timeseires>', so we can parse "
            f"it for all use cases. Current input: {data_filter}"
        )
    if split_type == "between_site":
        # get all the data
        tmp_data_list = load_h5_data_path(
            path=path, data_filter=data_filter, shuffle=False
        )
        # count number of sites
        all_datasets = [d.split("/")[1] for d in tmp_data_list]
        unique_datasets = set(all_datasets)
        datasets = list(unique_datasets)
        # train test split on the number of sites
        train, test = train_test_split(
            datasets, test_size=test_set, random_state=42
        )
        # concat file based on these info
        data_filter = data_filter.split("/")[-1]
        tng_data, test_data = [], []
        for i, split in enumerate([train, test]):
            for dset in split:
                cur_site_filter = f"{dset}/*/{data_filter}"
                data_list = load_h5_data_path(
                    path=path, task_filter=cur_site_filter, shuffle=True
                )
                if i == 0:
                    tng_data += data_list
                else:
                    test_data += data_list
        return tng_data, test_data

    elif split_type == "within_site":
        # for each site do a train test split
        # concatenate training / testing across sites
        data_list = load_h5_data_path(
            path=path, data_filter=data_filter, shuffle=False
        )  # shuffle in the train_test_split
        class_label = [d.split("/")[1] for d in data_list]
        tng_data, test_data = train_test_split(
            data_list,
            test_size=test_set,
            random_state=42,
            stratify=class_label,
        )
        return tng_data, test_data

    else:  # just shuffle everything
        data_list = load_h5_data_path(
            path=path, data_filter=data_filter, shuffle=False
        )  # shuffle in the train_test_split
        tng_data, test_data = train_test_split(
            data_list, test_size=test_set, random_state=42
        )
        return tng_data, test_data


def load_data(
    path: Union[Path, str],
    h5dset_path: Union[List[str], str],
    dtype: str = "data",
) -> List[Union[np.ndarray, str, int, float]]:
    """Load time series or phenotype data from the hdf5 files.

    Args:
        path (Union[Path, str]): Path to the hdf5 file.
        h5dset_path (Union[List[str], str]): Path to data inside the
            h5 file.
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
        return data_list
    else:
        with h5py.File(path, "r") as h5file:
            for p in h5dset_path:
                subject_node = "/".join(p.split("/")[:-1])
                data_list.append(h5file[subject_node].attrs[dtype])
        return data_list


def load_h5_data_path(
    path: Union[Path, str],
    data_filter: Union[str, None] = None,
    shuffle: bool = False,
) -> List[str]:
    """Load dataset path data from HDF5 file.

    Args:
      path (str): path to the HDF5 file
      data_filter (str): regular expression to apply on run names
        (default=None)
      shuffle (bool): whether to shuffle the data (default=False)

    Returns:
      (list of str): HDF5 path to data
    """
    data_list = []
    with h5py.File(path, "r") as h5file:
        for dset in _traverse_datasets(h5file):
            if data_filter is None or re.search(data_filter, dset):
                data_list.append(dset)
    if shuffle and data_list:
        import random

        random.seed(42)
        random.shuffle(data_list)
    return data_list


def _traverse_datasets(hdf_file):
    """Load nested hdf5 files.
    https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
    """  # ruff: noqa: W505

    def h5py_dataset_iterator(g, prefix=""):
        for key in g.keys():
            item = g[key]
            path = f"{prefix}/{key}"
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path
