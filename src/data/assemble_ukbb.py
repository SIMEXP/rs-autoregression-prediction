"""Select data based on abide QC results. Save as HDF5
This is a standalone script to assemble the ukbb dataset.
As ukbb has really short tr (0.735s), we are separating the data into
4 segments so it resembles data with TR = 3s.
I isolate this from the ML workflow as we don't need to rerun this step.
Dependencies noted in the requirements.txt file of this directory.
"""
import argparse
import fnmatch
import re
import tarfile
from pathlib import Path

import h5py
import pandas as pd
from tqdm import tqdm


def load_data(path, task_filter=None):
    """Load pre-processed data from HDF5 file.

    Args:
      path (str): path to the HDF5 file
      task_filter (str): regular expression to apply on run names
        (default=None)

    Returns:
      (list of numpy arrays): loaded data
    """
    data_list = {}
    with h5py.File(path, "r") as h5file:
        for dset in _traverse_datasets(h5file):
            if task_filter is None or re.search(task_filter, dset):
                data_list[dset] = h5file[dset][:]
    return data_list


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phenotype-file",
        type=str,
        help="Path to phenotype file file generated by wrangling-phenotype project.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inputs/connectomes/ukbb.h5",
        help="Path to save the output",
    )
    parser.add_argument(
        "--segment-tr",
        action="store_true",
        help="Segment the data into 4 segments (mimic TR=3). "
        "Each segemnt of data is saved alongside the original data with"
        "the `seg-{index}` entity.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to connectome archive (tar.gz) generated by giga connectome (version <0.5).",
    )
    args = parser.parse_args()
    print(args)
    connectome_archive = args.input
    phenotype_file = str(args.phenotype_file)
    path_h5 = str(args.output)
    output_pheno_path = path_h5.replace(".h5", "_phenotype.tsv")

    phenotype = pd.read_csv(phenotype_file, sep="\t")
    subjects = phenotype["participant_id"].tolist()
    print(
        f"Found {len(subjects)} subjects that passed QC with phenotype data."
    )

    valid_subject = []
    with tarfile.open(connectome_archive, "r") as f:
        for member in tqdm(f.getmembers()):
            if "sub" not in member.name:
                continue
            subj = int(member.name.split("sub-")[-1].split("_")[0])
            if subj not in subjects:  # skip subjects that did not pass QC
                continue

            if fnmatch.fnmatch(member.name, "*simple+gsr.h5"):
                valid_subject.append(subj)
                f.extract(member, path=".")
                path_tmp = Path(member.name)
                subj_data = load_data(path_tmp, task_filter="sub-.*timeseries")
                path_tmp.unlink()
                for k, v in subj_data.items():
                    dset_name = f"/ukbb/sub-{subj}{k}"
                    with h5py.File(path_h5, "a") as new_f:
                        new_f.create_dataset(dset_name, data=v)

                    for i in range(4):
                        seg_dset_name = dset_name.replace(
                            "timeseries", f"seg-{i+1}_timeseries"
                        )
                        with h5py.File(path_h5, "a") as new_f:
                            new_f.create_dataset(
                                seg_dset_name, data=v[i::4, :]
                            )
            else:
                continue
    with h5py.File(path_h5, "a") as new_f:
        new_f.create_dataset("/participant_id", data=valid_subject)

    print(f"data with connectome and phenotype data: {len(valid_subject)}")
    phenotype = phenotype.set_index("participant_id")
    phenotype = phenotype.loc[valid_subject, :]
    phenotype = phenotype.sort_index()

    # save the phenotype data as well
    phenotype.to_csv(output_pheno_path, sep="\t")

    # sanity check
    with h5py.File(path_h5, "r") as h5file:
        participant_id = list(h5file["ukbb"].keys())

    assert len(valid_subject) == phenotype.shape[0]
    assert len(valid_subject) == len(participant_id)


if __name__ == "__main__":
    main()
