import argparse
from pathlib import Path

import h5py
from sklearn.model_selection import train_test_split
from src.data import load_data


def main():
    """
    Create train test split.
    For the priliminary experiment:
    - Training: abide 1
    - Testing: abide 2

    Validation fold within training: randomly select 20% of data as validation.
    """
    path_dataset = Path("inputs/connectomes/abide1.h5")
    with h5py.File(path_dataset, "r") as f:
        h5dsets = list(load_data._traverse_datasets(f))
    train, validation = train_test_split(
        h5dsets, test_size=0.2, random_state=42
    )

    with h5py.File(path_dataset, "r") as f:
        for dset in load_data._traverse_datasets(f):
            if dset in train:
                with h5py.File("inputs/abide1_train.h5", "a") as f_train:
                    f_train.create_dataset(
                        dset,
                        data=f[dset][:],
                    )

    with h5py.File(path_dataset, "r") as f:
        for dset in load_data._traverse_datasets(f):
            if dset in validation:
                with h5py.File("inputs/abide1_validation.h5", "a") as f_val:
                    f_val.create_dataset(
                        dset,
                        data=f[dset][:],
                    )


if __name__ == "__main__":
    main()
