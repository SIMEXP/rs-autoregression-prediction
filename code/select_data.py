"""Select data based on abide QC results. Save as HDF5."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

field_mappers = {
    "abide1": {"site_name": "site_name", "SUB_ID": "SUB_ID"},
    "abide2": {"site_name": "SITE_ID", "SUB_ID": "SUB_ID"},
}

tr_mapper = {
    "abide1": 2500,
    "abide2": {
        "BNI_1": 3000,
        "EMC_1": 2000,
        "ETH_1": 2000,
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
        "TCD_1": 2000,
        "UCD_1": 2000,
        "UCLA_1": 3000,
        "USM_1": 2000,
        "UCLA_Long": 3000,
        "UPSM_Long": 1500,
    },
}

TARGET_TR = 2.5


def _traverse_datasets(hdf_file):
    """Load nested hdf5 files.
    https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
    """

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


def _parse_path(dset):
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


def _fetch_h5_group(f, subject, session):
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


def concat_h5(path_h5s, output_h5):
    """Concatenate connectome h5 files."""
    with h5py.File(output_h5, "a") as h5file:
        for p in path_h5s:
            print(p)
            site_name = p.parent.name
            site = h5file.create_group(site_name)
            with h5py.File(p, "r") as f:
                for dset in _traverse_datasets(f):
                    subject, session, dataset_name = _parse_path(dset)
                    data = f[dset][:]
                    if not subject and not session:
                        site.create_dataset(dataset_name, data=data)
                    else:
                        g = _fetch_h5_group(site, subject, session)
                        g.create_dataset(dataset_name, data=data)
    return output_h5


def _resample_tr(data, original_tr):
    if TARGET_TR == original_tr:
        return data
    time_stamp_original = np.arange(0, data.shape[0]) * original_tr
    scan_length_second = data.shape[0] * original_tr
    n_resampled_timepoints = int(scan_length_second / TARGET_TR)
    time_stamp_new = np.arange(0, n_resampled_timepoints) * TARGET_TR
    f = interp1d(time_stamp_original.T, data.T, fill_value="extrapolate")
    return f(time_stamp_new.T).T


def _process_data(abide_version, site_name, subjects, f, dset):
    subject, _, _ = _parse_path(dset)
    if not subject or "connectome" in dset:
        del f[dset]
        print(f"remove {dset}")
        return

    current_sub = int(subject.split("sub-")[-1])
    if current_sub not in subjects:
        print(current_sub)
        del f[dset]
        return

    print(f"resample {dset}")
    data = f[dset][:]
    # resample the time series
    resampled = _resample_tr(data, tr_mapper[abide_version][site_name])
    del f[dset]
    f.create_dataset(dset, data=resampled)
    return


def _get_subjects_passed_qc(qc, abide_dataset_name, site_name):
    site_filter = qc["site_name"] == site_name
    subjects = qc.loc[
        site_filter, field_mappers[abide_dataset_name]["SUB_ID"]
    ].values
    subjects = subjects.tolist()
    return subjects


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
        f"inputs/connectomes/sourcedata/{abide_version}_connectomes/"
    ).glob("*/atlas-MIST_desc-simple+gsr.h5")
    path_concat = f"inputs/connectomes/{abide_version}.h5"
    if not Path(path_concat).exists():
        path_concat = concat_h5(path_connectomes, path_concat)
    print("concatenated across site")

    # select subject based on each site, keep time series only and resample
    qc = pd.read_csv(
        f"inputs/connectomes/sourcedata/{abide_version}_connectomes/{abide_version.upper()}_Pheno_PSM_matched.tsv",
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
        with h5py.File(path_concat, "a") as f:
            for dset in _traverse_datasets(f):
                print(dset)
                _process_data(abide_version, site_name, subjects, f, dset)


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
    assert data_tr3.shape[1] == 122
    data_tr25 = _resample_tr(data, TARGET_TR)
    assert data_tr25.shape[1] == 122
    assert data_tr25.shape[0] == 198
