import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data_by_site(
    path: Path,
    data_filter: str,
    test_set: float = 0.25,
    split_type: Union[str, None] = None,
    datasets: Union[List[str], None] = None,
    random_state: int = 42,
) -> Tuple[List[np.ndarray]]:
    """Train-test split with different strategies for multiple datasets.

    Args:
        path (Path): Path to the h5 file with all the data.
        datasets (List[str]): a list of dataset names to process.
        test_set (float): proportion of the test set size.
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
        # train test split on the number of sites
        train, test = train_test_split(
            datasets, test_size=test_set, random_state=random_state
        )
        # concat file based on these info
        data_filter = data_filter.split("/")[-1]
        tng_data, test_data = [], []
        for i, split in enumerate([train, test]):
            for dset in split:
                cur_site_filter = f"{dset}/*/{data_filter}"
                data_list = load_h5_data_path(
                    path=path,
                    task_filter=cur_site_filter,
                    shuffle=True,
                    random_state=random_state,
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
            random_state=random_state,
            stratify=class_label,
        )
        return tng_data, test_data

    else:  # just shuffle everything
        data_list = load_h5_data_path(
            path=path, data_filter=data_filter, shuffle=False
        )  # shuffle in the train_test_split
        tng_data, test_data = train_test_split(
            data_list, test_size=test_set, random_state=random_state
        )
        return tng_data, test_data


def load_ukbb_dset_path(
    path: Union[Path, str],
    atlas_desc: str,
    n_sample: int = 50,
    val_set: float = 0.25,
    test_set: float = 0.25,
    segment: Union[int, List[int]] = -1,
    random_state: int = 42,
) -> Dict:
    """Load time series of UK Biobank.

    We segmented the time series per subject as independent samples,
    hence it's important to make sure the same subject is not in both
    training and testing set.

    Args:
        path (Union[Path, str]): Path to the hdf5 file.
        atlas_desc (str): Regex pattern to look for suitable data,
            such as the right `desc` field for atlas,
            e.g., "atlas-MIST_desc-197".
        n_sample (int, optional): number of subjects to use.
            Defaults to 50, and -1 would take the full sample.
        val_set (float, optional): proportion of the validation set
            size in relation to the full sample. Defaults to 0.25.
        test_set (float, optional): proportion of the test set size
            in relation to the full sample. Defaults to 0.25.
        segment (Union[int, List[int]], optional): segments of the
            time series to use. 0 for the full time series.
            Defaults to -1 to load all four segments.

    Returns:
        List[Union[np.ndarray, str, int, float]]: loaded data.
    """
    if isinstance(segment, int) and segment > 4:
        raise ValueError(
            "Segment number should be between 1 and 4, inclusive."
            f"Current input: {segment}"
        )
    if isinstance(segment, list) and any([s > 4 for s in segment]):
        raise ValueError(
            "Segment number should be between 1 and 4, inclusive."
            f"Current input: {segment}"
        )

    if segment == -1:
        segment = [1, 2, 3, 4]
    if segment == 0:
        segment = [None]
    elif segment <= 4:
        segment = [segment]

    # get the participant IDs to use
    with h5py.File(path, "r") as h5file:
        participant_id = list(h5file["ukbb"].keys())

    if n_sample == -1:
        pass
    elif n_sample < len(participant_id):
        total_proportion_sample = n_sample / len(participant_id)
        participant_id, _ = train_test_split(
            participant_id,
            test_size=(1 - total_proportion_sample),
            random_state=random_state,
        )

    # construct path
    subject_path_template = (
        "/ukbb/{sub}/{sub}_task-rest_{atlas_desc}_{seg}timeseries"
    )
    data_list = []
    for sub in participant_id:
        for seg in segment:
            seg = f"seg-{seg}_" if seg is not None else ""
            cur_sub_path = subject_path_template.format(
                sub=sub, seg=seg, atlas_desc=atlas_desc
            )
            data_list.append(cur_sub_path)
    # train-test-val split
    train, test = train_test_split(
        data_list, test_size=test_set, random_state=random_state
    )
    # calculate the proportion of val_set in the training loop
    train, val = train_test_split(
        train, test_size=val_set / (1 - test_set), random_state=random_state
    )
    return {"train": train, "val": val, "test": test}


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


def load_h5_data_path(
    path: Union[Path, str],
    data_filter: Union[str, None] = None,
    shuffle: bool = False,
    random_state: int = 42,
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
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(data_list)
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


def get_model_data(
    data_file: Union[Path, str],
    dset_path: List[str],
    phenotype_file: Union[Path, str],
    measure: str = "connectome",
    label: str = "sex",
    pooling_target: str = "max",
    log: logging = logging,
) -> Dict[str, np.ndarray]:
    """Get the data from pretrained model for the downstrean task.

    Args:
        data_file (Union[Path, str]): Path to the hdf5 file.
        dset_path (List[str]): List of path to the data inside the
            h5 file.
        phenotype_file (Union[Path, str]): Path to the phenotype file.
        measure (str, optional): Measure to use for the data. Defaults
            to "connectome".
        label (str, optional): Label to use for the data. Defaults to
            "sex".
        log (logging): logging object.

    Returns:
        Dict[str, np.ndarray]: Dictionary with the data and label.

    Raises:
        NotImplementedError: If the measure is not supported.
    """

    if measure not in [
        "connectome",
        "r2map",
        "avgr2",
        "conv_max",
        "conv_avg",
        "conv_std",
    ]:
        raise NotImplementedError(
            "measure must be one of 'connectome', 'r2map', 'avgr2'"
        )
    if measure == "connectome":
        cm = ConnectivityMeasure(kind="correlation", vectorize=True)

    participant_id = [
        p.split("/")[-1].split("sub-")[-1].split("_")[0] for p in dset_path
    ]
    n_total = len(participant_id)
    df_phenotype = load_phenotype(phenotype_file)
    # get the common subject in participant_id and df_phenotype
    participant_id = list(set(participant_id) & set(df_phenotype.index))
    # get extra subjects in participant_id
    # remove extra subjects in dset_path
    log.info(
        f"Subjects with phenotype data: {len(participant_id)}. Total subjects: {n_total}"
    )

    dataset = {}
    for p in dset_path:
        subject = p.split("/")[-1].split("sub-")[-1].split("_")[0]
        if subject in participant_id:
            df_phenotype.loc[subject, "path"] = p
    selected_path = df_phenotype.loc[participant_id, "path"].values.tolist()
    log.info(len(selected_path))
    data = load_data(data_file, selected_path, dtype="data")

    if "r2" in measure:
        data = np.concatenate(data).squeeze()
        if measure == "avgr2":
            data = data.mean(axis=1).reshape(-1, 1)
        data = StandardScaler().fit_transform(data)

    if measure == "connectome":
        data = cm.fit_transform(data)

    if "conv" in measure:
        data = np.array(data)
        data = StandardScaler().fit_transform(data)

    labels = df_phenotype.loc[participant_id, label].values
    log.info(f"data shape: {data.shape}")
    log.info(f"label shape: {labels.shape}")
    dataset = {"data": data, "label": labels}
    return dataset


def load_phenotype(path: Union[Path, str]) -> pd.DataFrame:
    """Load the phenotype data from the file.

    Args:
        path (Union[Path, str]): Path to the phenotype file.

    Returns:
        pd.DataFrame: Phenotype data.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"participant_id": str, "age": float, "sex": str, "site": str},
    )
    df = df.set_index("participant_id")
    return df
