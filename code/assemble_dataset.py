"""Select data based on abide QC results. Save as HDF5."""
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from nilearn import signal
from scipy.interpolate import interp1d
from src.data import load_data
from tqdm import tqdm

# recode all baseline as zero
FIELD_MAPPERS = {
    "abide1": {  # http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf
        "site_name": {"column": "site_name"},
        "participant_id": {"column": "SUB_ID"},
        "sex": {"column": "SEX", "value": {1: 1, 2: 0}},  # male  # female
        "age": {"column": "AGE_AT_SCAN"},
        "diagnosis": {
            "column": "DX_GROUP",
            "value": {2: 0, 1: 1},  # control  # autism
        },
    },
    "abide2": {  # http://fcon_1000.projects.nitrc.org/indi/abide/ABIDEII_Data_Legend.pdf
        "site_name": {"column": "SITE_ID"},
        "participant_id": {"column": "SUB_ID"},
        "sex": {"column": "SEX", "value": {"male": 1, "female": 0}},
        "age": {"column": "AGE_AT_SCAN"},
        "diagnosis": {
            "column": "DX_GROUP",
            "value": {"Control": 0, "Autism": 1},  # control  # autism
        },
    },
}

TR_MAPPERS = {
    "abide1": {
        "Caltech": 2000,
        "CMU_a": 2000,
        "CMU_b": 2000,
        "KKI": 2500,
        "Leuven_1": 1656,
        "Leuven_2": 1667,
        "MaxMun_a": 3000,
        "MaxMun_b": 3000,
        "MaxMun_c": 3000,
        "MaxMun_d": 3000,
        "NYU": 2000,
        "Olin": 1500,
        "OHSU": 2500,
        "Pitt": 1500,
        "SDSU": 2000,
        "SBL": 2200,
        "Stanford": 2000,
        "Trinity": 2000,
        "UCLA_1": 3000,
        "UCLA_2": 3000,
        "UM_1": 2000,
        "UM_2": 2000,
        "USM": 2000,
        "Yale": 2000,
    },
    "abide2": {
        "BNI_1": 3000,
        "EMC_1": 2000,
        "ETHZ_1": 2000,
        "GU_1": 2000,
        "IU_1": 813,
        "IP_1": 2700,
        "KKI_1": 2500,
        "KUL_3": 2500,
        "NYU_1": 2000,
        "NYU_2": 2000,
        "OHSU_1": 2500,
        "ONRC_2": 475,
        "SDSU_1": 2000,
        "SU_2": 2500,  # double check
        "TCD_1": 2000,
        "UCD_1": 2000,
        "UCLA_1": 3000,
        "USM_1": 2000,
        "UCLA_Long": 3000,
        "UPSM_Long": 1500,
        "U_MIA_1": 2500,  # double check
    },
}

TARGET_TR = 2.5


def _parse_path(
    dset: str,
) -> Tuple[Union[str, None], Union[str, None], Union[str, None]]:
    """Get subject, session, dataset name from a BIDS path.

    Args:
        dset (str): h5 dataset path.

    Returns:
        Tuple[Union[str, None], Union[str, None], Union[str, None]]:
        A tuple containing subject, session, and dataset name.
        The value can be None if it's not present in path.
    """
    dataset_name = dset.split("/")[-1]
    if "sub" in dataset_name:
        subject = f"sub-{dataset_name.split('sub-')[-1].split('_')[0]}"
    else:
        subject = None
    if "ses" in dset:
        session = f"ses-{dataset_name.split('ses-')[-1].split('_')[0]}"
    else:
        session = None
    return subject, session, dataset_name


def _fetch_h5_group(
    f: h5py.File, subject: str, session: str = None
) -> h5py.Group:
    """Get group from a HDF5 file containing a BIDS-ish structure.

    Args:
        f (h5py.File): HDF5 file object of the connectomes.
        subject (str): BIDS subject ID includes the entity "sub-".
        session (str, optional): Session name

    Returns:
        h5py.Group: H5 group matching the BIDS information
    """
    if subject not in f:
        return (
            f.create_group(f"{subject}/{session}")
            if session
            else f.create_group(f"{subject}")
        )
    elif session:
        return (
            f[f"{subject}"].create_group(f"{session}")
            if session not in f[f"{subject}"]
            else f[f"{subject}/{session}"]
        )
    else:
        return f[f"{subject}"]


def concat_h5(path_h5s: Path, output_h5: Path) -> Path:
    """Concatenate connectome h5 files.

    Args:
        path_h5s (Path): Paths to HDF5 files to concatenate.
        output_h5 (Path): Output HDF5 file path.

    Returns:
        Path: output HDF5 file path, concatenated.
    """
    with h5py.File(output_h5, "a") as h5file:
        for p in path_h5s:
            site_name = p.parent.name
            site = h5file.create_group(site_name)
            with h5py.File(p, "r") as f:
                for dset in tqdm(load_data._traverse_datasets(f)):
                    subject, session, dataset_name = _parse_path(dset)
                    data = f[dset][:]
                    if subject or session:
                        g = _fetch_h5_group(site, subject, session)
                        g.create_dataset(dataset_name, data=data)
                    else:
                        site.create_dataset(dataset_name, data=data)
    return output_h5


def _resample_tr(data: np.ndarray, original_tr: float) -> np.ndarray:
    """Resample time series data to a set TR.

    Args:
        data (np.ndarray): the time series data
        original_tr (float): TR of the original data in seconds

    Returns:
        np.ndarray: Resampled timeseries data in TR=2.5
    """
    # standardise to percent signal change
    # data was already detrended and z scored....

    # resample data
    if TARGET_TR == original_tr:
        return data
    time_stamp_original = np.arange(0, data.shape[0]) * original_tr
    scan_length_second = data.shape[0] * original_tr
    n_resampled_timepoints = int(scan_length_second / TARGET_TR)
    time_stamp_new = np.arange(0, n_resampled_timepoints) * TARGET_TR
    f = interp1d(time_stamp_original.T, data.T, fill_value="extrapolate")
    resampled = f(time_stamp_new).T

    # make sure the value is somehow standardised
    resampled = signal.clean(resampled, detrend=False, standardize=True)
    return resampled


def _get_subjects_passed_qc(
    qc: pd.DataFrame, abide_dataset_name: str, site_name: str
) -> dict:
    """Get ABIDE subjects that passed quality control from Urch's work.
    Also gather metadata

    Args:
        qc (pd.DataFrame): quality control results.
        abide_dataset_name (str): abide1 or abide2
        site_name (str): abide site name

    Returns:
        dict: subject ID as keys and values: age, sex, diagnosis, site.
    """
    site_filter = qc["site_name"] == site_name
    subjects = qc.loc[
        site_filter,
        FIELD_MAPPERS[abide_dataset_name]["participant_id"]["column"],
    ].values.tolist()

    subjects_info = {}
    for subject in subjects:
        subject_filter = (
            qc[FIELD_MAPPERS[abide_dataset_name]["participant_id"]["column"]]
            == subject
        )
        if abide_dataset_name == "abide1":
            # zero padded
            subject = f"{subject:07}"

        subjects_info[str(subject)] = {
            "dataset": abide_dataset_name,
            "site": site_name,
        }
        for field in ["sex", "age", "diagnosis"]:
            original_value = qc.loc[
                subject_filter,
                FIELD_MAPPERS[abide_dataset_name][field]["column"],
            ].values.tolist()[0]
            value_mapper = FIELD_MAPPERS[abide_dataset_name][field].get(
                "value", False
            )
            if value_mapper:
                subjects_info[str(subject)][field] = value_mapper[
                    original_value
                ]
            else:
                subjects_info[str(subject)][field] = original_value
    return subjects_info


def _check_subject_pass_qc(dset: str, subjects: List[str]) -> bool:
    """Check if the current subjects passed quality control.

    Args:
        dset (str): Current H5 dataset path.
        subjects (List[str]): List of subjects that passed QC

    Returns:
        bool: If the current subject passed QC.
    """
    subject, _, _ = _parse_path(dset)
    if not subject or "connectome" in dset:
        return False
    else:
        current_sub = subject.split("sub-")[-1]
    return current_sub in subjects


def _get_abide_tr(abide_version: str, site_name: str) -> float:
    """Get ABIDE dataset TR (in milliseconds) in seconds.

    Args:
        abide_version (str): abide1 or abide2
        site_name (str): name of abide sites

    Returns:
        float: TR in seconds
    """
    original_tr = TR_MAPPERS[abide_version]
    if not isinstance(original_tr, int):
        original_tr = original_tr[site_name]
    original_tr /= 1000
    return original_tr


def main():
    site_list = []
    for abide_version in FIELD_MAPPERS:
        path_connectomes = Path(
            f"inputs/connectomes/sourcedata/{abide_version}/"
        ).glob("*_connectomes-0.*/*/atlas-MIST_desc-simple+gsr.h5")
        path_concat = "inputs/connectomes/processed_connectomes.h5"
        path_tmp = Path(f"{abide_version}_tmp.h5")
        if not Path(path_tmp).exists():
            path_tmp = concat_h5(path_connectomes, path_tmp)
        print("concatenated across site")

        # select subject based on each site,
        # keep time series only and resample
        qc = pd.read_csv(
            f"inputs/connectomes/sourcedata/{abide_version}/{abide_version.upper()}_Pheno_PSM_matched.tsv",
            sep="\t",
        )
        qc["site_name"] = qc[
            FIELD_MAPPERS[abide_version]["site_name"]["column"]
        ].replace({"ABIDEII-": ""}, regex=True)
        all_sites = qc["site_name"].unique()
        print(f"found {qc.shape[0]} subjects passed QC.")

        for site_name in all_sites:
            print(site_name)
            subjects_info = _get_subjects_passed_qc(
                qc, abide_version, site_name
            )
            subjects = list(subjects_info.keys())
            if len(subjects) == 0:
                print("found no valid subjects.")
                continue
            original_tr = _get_abide_tr(abide_version, site_name)

            with h5py.File(path_tmp, "r") as f:
                for dset in tqdm(load_data._traverse_datasets(f)):
                    if not _check_subject_pass_qc(dset, subjects):
                        continue
                    # resample the time series
                    data = f[dset][:]
                    resampled = _resample_tr(data, original_tr)
                    dset_name = dset.split("/")[-1]
                    participant_id = dset_name.split("_")[0].split("sub-")[-1]
                    dset_name = f"{abide_version}_site-{site_name}/sub-{participant_id}/{dset_name}"
                    # save data
                    with h5py.File(path_concat, "a") as new_f:
                        new_f.create_dataset(dset_name, data=resampled)

            # populate phenotype data
            with h5py.File(path_concat, "a") as new_f:
                for participant_id in tqdm(subjects_info):
                    node = f"{abide_version}_site-{site_name}/sub-{participant_id}"
                    if node in new_f:
                        print("phenotype data")
                        dset_sub = new_f[node]
                        phenotype = subjects_info[participant_id]
                        for field in phenotype:
                            dset_sub.attrs.create(
                                name=field, data=phenotype[field]
                            )

            # dataset metadata
            with h5py.File(path_concat, "a") as f:
                node_site = f"{abide_version}_site-{site_name}"
                if node in f:
                    site_list.append(node_site)
                    print("dataset meta data")
                    node_site = f[node_site]
                    node_site.attrs["dataset_name"] = abide_version
                    node_site.attrs["diagnosis_name"] = "autism"
                    node_site.attrs["diagnosis_code_control"] = 0
                    node_site.attrs["diagnosis_code_patient"] = 1
                    node_site.attrs["sex_male"] = 1
                    node_site.attrs["sex_female"] = 0

        print("add full dataset list to attribute")
        with h5py.File(path_concat, "a") as f:
            f.attrs["dataset_list"] = site_list
            f.attrs["complied_date"] = str(datetime.today())
        # delete the temporary file
        # path_tmp.unlink()


if __name__ == "__main__":
    main()


def test_parse_path():
    s, ss, name = _parse_path("/atlas-blah_desc-400_connectome")
    assert s is None
    assert ss is None
    assert name == "atlas-blah_desc-400_connectome"

    s, ss, name = _parse_path(
        "/sub-001/ses-1/sub-001_ses-1_task-rest_atlas-blah_desc-400_connectome"
    )
    assert s == "sub-001"
    assert ss == "ses-1"
    assert name == "sub-001_ses-1_task-rest_atlas-blah_desc-400_connectome"


def test_resample_tr():
    # time x parcells
    data = np.random.random((198, 7))
    data_tr3 = _resample_tr(data, 3)
    scan_length_second = 198 * 3
    n_resampled_timepoints = int(scan_length_second / TARGET_TR)
    assert data_tr3.shape[0] == n_resampled_timepoints
    assert data_tr3.shape[1] == 7
    assert (np.std(data_tr3[:, 0]) - 1) < 1.0e-06  # zscore

    data_tr25 = _resample_tr(data, TARGET_TR)
    assert data_tr25.shape[1] == 7
    assert data_tr25.shape[0] == 198
    assert (np.std(data_tr25[:, 0]) - 1) < 1.0e-06  # zscore
