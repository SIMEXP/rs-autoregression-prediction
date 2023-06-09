"""Select data based on abide QC results. Save as HDF5."""
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
    "abide1": {"CMU_a": 2.5},
    "abide2": {"BNI_1": 2.5},
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
    subject = f"sub-{dataset_name.split('sub-')[-1].split('_')[0]}"
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
    # with tb.File('table_copy_2.h5',mode='w') as h5fw:
    #     for h5name in glob.glob('file*.h5'):
    #         h5fr = tb.File(h5name,mode='r')
    #         print (h5fr.root._v_children)
    #     h5fr.root._f_copy_children(h5fw.root)
    with h5py.File(output_h5, "a") as h5file:
        for p in path_h5s:
            print(p)
            site_name = p.parent.name
            site = h5file.create_group(site_name)
            with h5py.File(p, "r") as f:
                for dset in _traverse_datasets(f):
                    subject, session, dataset_name = _parse_path(dset)
                    g = _fetch_h5_group(site, subject, session)
                    data = f[dset][:]
                    # save data
                    if "rest" in dataset_name:
                        g.create_dataset(dataset_name, data=data)
                    else:
                        site.create_dataset(dataset_name, data=data)
    return output_h5


def _resample_tr(data, original_tr):
    if TARGET_TR == original_tr:
        return data
    time_stamp_original = np.arange(0, data.shape[0]) * original_tr
    scan_length = data.shape[0] * data.shape[0]
    n_resampled_timepoints = int(scan_length / TARGET_TR)
    time_stamp_new = np.arange(0, n_resampled_timepoints) * TARGET_TR
    f = interp1d(time_stamp_original, data)
    return f(time_stamp_new)


def _process_data(site_name, subjects, f, dset):
    if "sub" not in dset or "connectome" in dset:
        del f[dset]
        print(f"remove {dset}")
    subject, _, _ = _parse_path(dset)
    print(subject)
    current_sub = subject.split("sub-")[-1]
    if current_sub != "atlas-MIST":
        current_sub = int(subject.split("sub-")[-1])
    if current_sub == "atlas-MIST":
        pass
    elif current_sub not in subjects:
        del f[dset]
        print(f"remove {dset}")
    else:
        print(f"resample {dset}")
        data = f[dset][:]
        # resample the time series
        f[dset][:] = _resample_tr(data, tr_mapper[site_name])


def main():
    for key in field_mappers:
        path_connectomes = Path(
            f"inputs/connectomes/sourcedata/{key}_connectomes/"
        ).glob("*/atlas-MIST_desc-simple+gsr.h5")
        path_concat = f"inputs/connectomes/{key}.h5"
        if not Path(path_concat).exists():
            path_concat = concat_h5(path_connectomes, path_concat)
        print("concatenated across site")

        # select subject based on each site, keep time series only and resample
        qc = pd.read_csv(
            f"inputs/connectomes/sourcedata/{key}_connectomes/{key.upper()}_Pheno_PSM_matched.tsv",
            sep="\t",
        )
        qc["site_name"] = qc[field_mappers[key]["site_name"]].replace(
            {"ABIDEII-": ""}, regex=True
        )
        all_sites = qc["site_name"].unique()

        for site_name in all_sites:
            print(site_name)
            site_filter = qc["site_name"] == site_name
            subjects = qc.loc[site_filter, field_mappers[key]["SUB_ID"]].values
            if len(subjects) == 0:
                continue

            with h5py.File(path_concat, "a") as f:
                for dset in _traverse_datasets(f):
                    _process_data(site_name, subjects, f, dset)


if __name__ == "__main__":
    main()
