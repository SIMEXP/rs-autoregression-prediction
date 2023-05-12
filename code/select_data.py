"""Select data based on QC results. Save as HDF5."""
from pathlib import Path
import h5py
import pandas as pd


def _traverse_datasets(hdf_file):
    """Load nested hdf5 files.
    https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
    """
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
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


if __name__ == "__main__":
    qc_abide1 = pd.read_csv("inputs/connectomes/sourcedata/abide1_connectomes/ABIDE1_Pheno_PSM_matched.tsv", sep='\t', index_col=0)

    path_abide1 = Path("inputs/connectomes/sourcedata/abide1_connectomes/").glob("*/atlas-MIST_desc-simple+gsr.h5")


    with h5py.File("inputs/connectomes/abide1.h5", "a") as h5file:
        for p in path_abide1:
            site_name = p.parent.name
            site_filter = qc_abide1["site_name"] == site_name
            subjects = qc_abide1.loc[site_filter, "SUB_ID"].values
            if len(subjects) > 0:
                site = h5file.create_group(site_name)
                with h5py.File(p, "r") as f:
                    for dset in _traverse_datasets(f):
                        if "sub" in dset:
                            subject, session, dataset_name = _parse_path(dset)
                            current_sub = int(subject.split("sub-00")[-1])
                            if current_sub in subjects:
                                g = _fetch_h5_group(f, subject, session)
                                data = f[dset][:]
                                # save data
                                g.create_dataset(dataset_name, data=data)