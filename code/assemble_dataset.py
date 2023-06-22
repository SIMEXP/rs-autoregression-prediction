"""Select data based on abide QC results. Save as HDF5."""
import argparse
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from nilearn import signal
from scipy.interpolate import interp1d
from src.data import load_data

field_mappers = {
    "abide1": {"site_name": "site_name", "SUB_ID": "SUB_ID"},
    "abide2": {"site_name": "SITE_ID", "SUB_ID": "SUB_ID"},
}

tr_mapper = {
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
) -> Tuple[Union(str, None), Union(str, None), Union(str, None)]:
    """Get subject, session, dataset name from a BIDS path.

    Args:
        dset (str): h5 dataset path.

    Returns:
        Tuple[Union(str, None), Union(str, None), Union(str, None)]:
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
                for dset in load_data._traverse_datasets(f):
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
    # data was already detrended.
    data_psc = signal.clean(data, detrend=False, standardize="psc")
    data_psc /= 100  # make sure the value is between 0 and +-1

    # resample data
    if TARGET_TR == original_tr:
        return data_psc
    time_stamp_original = np.arange(0, data_psc.shape[0]) * original_tr
    scan_length_second = data_psc.shape[0] * original_tr
    n_resampled_timepoints = int(scan_length_second / TARGET_TR)
    time_stamp_new = np.arange(0, n_resampled_timepoints) * TARGET_TR
    f = interp1d(time_stamp_original.T, data_psc.T, fill_value="extrapolate")
    return f(time_stamp_new.T).T


def _get_subjects_passed_qc(
    qc: pd.DataFrame, abide_dataset_name: str, site_name: str
) -> List[str]:
    """Get ABIDE subjects that passed quality control from Urch's work.

    Args:
        qc (pd.DataFrame): quality control results.
        abide_dataset_name (str): abide1 or abide2
        site_name (str): abide site name

    Returns:
        List[str]: subject ID, passed quality control.
    """
    site_filter = qc["site_name"] == site_name
    subjects = qc.loc[
        site_filter, field_mappers[abide_dataset_name]["SUB_ID"]
    ].values
    return [int(s) for s in subjects.tolist()]


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
        current_sub = int(subject.split("sub-")[-1])
    return current_sub in subjects


def _get_abide_tr(abide_version: str, site_name: str) -> float:
    """Get ABIDE dataset TR (in milliseconds) in seconds.

    Args:
        abide_version (str): abide1 or abide2
        site_name (str): name of abide sites

    Returns:
        float: TR in seconds
    """
    original_tr = tr_mapper[abide_version]
    if not isinstance(original_tr, int):
        original_tr = original_tr[site_name]
    original_tr /= 1000
    return original_tr


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Concatenate abide sites into one dataset, select based on QC,"
            "Resample the time series to TR = 2.5."
        ),
    )
    parser.add_argument(
        "abide_version",
        help="Select abide1 or abide2",
        choices=["abide1", "abide2"],
    )

    args = parser.parse_args()
    abide_version = args.abide_version

    path_connectomes = Path(
        f"inputs/connectomes/sourcedata/{abide_version}/{abide_version}_connectomes-0.2.0/"
    ).glob("*/atlas-MIST_desc-simple+gsr.h5")
    path_concat = f"inputs/connectomes/{abide_version}.h5"
    path_tmp = Path("tmp.h5")
    if not Path(path_tmp).exists():
        path_tmp = concat_h5(path_connectomes, path_tmp)
    print("concatenated across site")

    # select subject based on each site,
    # keep time series only and resample
    qc = pd.read_csv(
        f"inputs/connectomes/sourcedata/{abide_version}/{abide_version.upper()}_Pheno_PSM_matched.tsv",
        sep="\t",
    )
    qc["site_name"] = qc[field_mappers[abide_version]["site_name"]].replace(
        {"ABIDEII-": ""}, regex=True
    )
    all_sites = qc["site_name"].unique()

    for site_name in all_sites:
        print(site_name)
        subjects = _get_subjects_passed_qc(qc, abide_version, site_name)
        if len(subjects) == 0:
            continue
        original_tr = _get_abide_tr(abide_version, site_name)

        with h5py.File(path_tmp, "r") as f:
            for dset in load_data._traverse_datasets(f):
                if not _check_subject_pass_qc(dset, subjects):
                    continue
                data = f[dset][:]
                # resample the time series,
                resampled = _resample_tr(data, original_tr)
                with h5py.File(path_concat, "a") as new_f:
                    new_f.create_dataset(
                        f"site-{site_name}_{dset.split('/')[-1]}",
                        data=resampled,
                    )
    # remove the temporary file
    path_tmp.unlink()


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
    data = np.random.random((198, 122))
    data_tr3 = _resample_tr(data, 3)
    scan_length_second = 198 * 3
    n_resampled_timepoints = int(scan_length_second / TARGET_TR)
    assert data_tr3.shape[0] == n_resampled_timepoints
    assert data_tr3.shape[1] == 122
    data_tr25 = _resample_tr(data, TARGET_TR)
    assert data_tr25.shape[1] == 122
    assert data_tr25.shape[0] == 198
