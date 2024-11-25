import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from general_class_balancer import general_class_balancer as gcb
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from src.data.utils import (
    TimeSeriesDataset,
    create_connectome,
    load_data,
    make_sequence_single_subject,
)
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

log = logging.getLogger(__name__)


class UKBBDataModule(LightningDataModule):
    """`LightningDataModule` for UK Biobank dataset from giga processing.

    This will only work for internally processed UK Biobank data.
    ```
    """

    def __init__(
        self,
        connectome_file: str,
        phenotype_file: str,
        data_dir: str = "data/",
        atlas: Tuple[str, int] = ("MIST", 197),
        timeseries_decimate: int = 4,
        timeseries_window_stride_lag: Tuple[int, int, int] = (16, 1, 1),
        timeseries_horizon: int = 6,
        train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
        class_balance_confounds: List[str] = (
            "site",
            "sex",
            "age",
            "mean_fd_raw",
            "proportion_kept",
        ),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        random_state: int = 1,
    ) -> None:
        """Initialize a `UKBBDataModule`."""
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.ToTensor()]
        # )
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        path_sample_split = (
            Path(self.hparams.data_dir)
            / f"downstream_sample_seed-{self.hparams.random_state}.json"
        )
        if not path_sample_split.exists():
            phenotype_meta = self.hparams.phenotype_file.replace(
                ".tsv", ".json"
            )
            new_sample = create_hold_out_sample(
                phenotype_path=self.hparams.phenotype_file,
                phenotype_meta=phenotype_meta,
                class_balance_confounds=self.hparams.class_balance_confounds,
                train_val_test_split=self.hparams.train_val_test_split,
                random_state=self.hparams.random_state,
            )
            # save matched sample to disk
            with open(path_sample_split, "w") as f:
                json.dump(new_sample, f, indent=2)
            del new_sample
        # load the existing sample for generating sequences
        with open(path_sample_split, "r") as f:
            sample = json.load(f)
        # h5 path
        out_path = Path(self.hparams.data_dir) / (
            f"atlas-{self.hparams.atlas[0]}{self.hparams.atlas[1]}_"
            f"decimate-{self.hparams.timeseries_decimate}_"
            f"windowsize-{self.hparams.timeseries_window_stride_lag[0]}_"
            f"stride-{self.hparams.timeseries_window_stride_lag[1]}_"
            f"lag-{self.hparams.timeseries_window_stride_lag[-1]}_"
            f"seed-{self.hparams.random_state}_data.h5"
        )
        if out_path.exists():
            return None

        conn_dset_paths = load_ukbb_dset_path(
            participant_id=sample["train"],
            atlas_desc=f"atlas-{self.hparams.atlas[0]}_desc-{self.hparams.atlas[1]}",
            segment=0,  # use full time series, decimate later in make_sequence_single_subject
        )
        # calculate connectome
        connectome = create_connectome(
            data_file=self.hparams.connectome_file,
            dset_paths=conn_dset_paths,
        )
        del conn_dset_paths
        # save connectome to disk as h5
        w, s, lag = self.hparams.timeseries_window_stride_lag
        m = self.hparams.timeseries_decimate  # decimate factor
        n_regions = self.hparams.atlas[1]
        with h5py.File(out_path, "a") as f:
            f.create_dataset("connectome", data=connectome)
            # make sequence for train, val, test, for model training
            for dset in sample:
                cur_group = f.create_group(dset)
                dset_paths = load_ukbb_dset_path(
                    participant_id=sample[dset],
                    atlas_desc=f"atlas-{self.hparams.atlas[0]}_desc-{self.hparams.atlas[1]}",
                    segment=0,  # always load the full time series, decimate later
                )
                log.info(f"Creating labels for {len(dset_paths)} scans.")
                for dset_path in tqdm(dset_paths):
                    data = load_data(self.hparams.connectome_file, dset_path)[
                        0
                    ]
                    x, y = make_sequence_single_subject(data, m, w, s, lag)
                    if x.shape[0] == 0 or x is None:
                        log.warning(
                            f"Skipping {dset} as label couldn't be created."
                        )
                        continue
                    if cur_group.get("input") is None:
                        cur_group.create_dataset(
                            name="input",
                            data=x,
                            dtype=np.float32,
                            maxshape=(None, n_regions, w),
                            chunks=(x.shape[0], n_regions, w),
                        )
                        cur_group.create_dataset(
                            name="label",
                            data=y,
                            dtype=np.float32,
                            maxshape=(None, n_regions),
                            chunks=(y.shape[0], n_regions),
                        )
                    else:
                        cur_group["input"].resize(
                            (cur_group["input"].shape[0] + x.shape[0]), axis=0
                        )
                        cur_group["input"][-x.shape[0] :] = x

                        cur_group["label"].resize(
                            (cur_group["label"].shape[0] + y.shape[0]), axis=0
                        )
                        cur_group["label"][-y.shape[0] :] = y

            # cur_group = f.create_group("test_horizon")
            # test_dsets = load_ukbb_dset_path(
            #     participant_id=sample["test"],
            #     atlas_desc=f"atlas-{self.hparams.atlas[0]}_desc-{self.hparams.atlas[1]}",
            #     segment=0,  # always load the full time series, decimate later
            # )
            # for dset_path in tqdm(test_dsets):
            #     fname = dset_path.split("/")[-1]
            #     cur_seg = cur_group.create_group(fname)
            #     with h5py.File(self.hparams.connectome_file, "r") as h5file:
            #         data = h5file[dset_path][:]
            #     X, _ = make_sequence_single_subject(
            #         data,
            #         m=m,
            #         length=w + self.hparams.timeseries_horizon,
            #         stride=s,
            #         lag=0,
            #     )
            #     del data
            #     Y = X[:, :, w:]
            #     cur_seg.create_dataset(name="input", data=X, dtype=np.float32)
            #     cur_seg.create_dataset(name="label", data=Y, dtype=np.float32)

    def setup(self, stage: str) -> None:
        """Set up the data for training, validation, and testing."""
        self.time_sequence_file = str(
            Path(self.hparams.data_dir)
            / (
                f"atlas-{self.hparams.atlas[0]}{self.hparams.atlas[1]}_"
                f"decimate-{self.hparams.timeseries_decimate}_"
                f"windowsize-{self.hparams.timeseries_window_stride_lag[0]}_"
                f"stride-{self.hparams.timeseries_window_stride_lag[1]}_"
                f"lag-{self.hparams.timeseries_window_stride_lag[-1]}_"
                f"seed-{self.hparams.random_state}_data.h5"
            )
        )
        time_sequence_h5 = h5py.File(self.time_sequence_file, "r")

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = TimeSeriesDataset(
                time_sequence_h5, set_type="train"
            )
            self.data_val = TimeSeriesDataset(
                time_sequence_h5,
                set_type="validation",
            )
            self.data_test = TimeSeriesDataset(
                time_sequence_h5,
                set_type="test",
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


def load_ukbb_dset_path(
    participant_id: List[str],
    atlas_desc: str,
    segment: Union[int, List[int]] = 0,
) -> List[str]:
    """Load time series path in h5 file of UK Biobank.

    We segmented the time series per subject as independent samples,
    hence it's important to make sure the same subject is not in both
    training and testing set.

    Args:
        participant_id List[str]: List of participant ID, excluding sub-.
        atlas_desc (str): Regex pattern to look for suitable data,
            such as the right `desc` field for atlas,
            e.g., "atlas-MIST_desc-197".
        segment (Union[int, List[int]], optional): segments of the
            time series to use. Default 0 for the full time series.
            -1 to load all four segments.

    Returns:
        List[str]: loaded data paths.
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
    elif segment == 0:
        segment = [None]
    elif isinstance(segment, int) and segment <= 4:
        segment = [segment]

    # construct path
    subject_path_template = (
        "/ukbb/sub-{sub}/sub-{sub}_task-rest_{atlas_desc}_{seg}timeseries"
    )
    h5_path = []
    for sub in participant_id:
        for seg in segment:
            seg = f"seg-{seg}_" if seg is not None else ""
            cur_sub_path = subject_path_template.format(
                sub=sub, seg=seg, atlas_desc=atlas_desc
            )
            h5_path.append(cur_sub_path)
    return h5_path


def load_ukbb_sets(
    data_dir: str, seed: int, atlas: Tuple[str, int], stage="test"
) -> List[str]:
    """Load train / validation / test set h5 dataset path for ukbb.

    Args:
        data_dir (str): _description_
        seed (int): _description_
        atlas (Tuple[str, int]): _description_
        stage (str, optional): _description_. Defaults to "test".

    Returns:
        Dict: _description_
    """
    if stage == "test_downstreams":
        raise ValueError(
            "Request the label of patient group: 'ADD', "
            "'ALCO', 'DEP', 'SCZ', 'BIPOLAR', 'PARK', 'MS', 'EPIL'"
        )
    # get dset paths
    path_sample_split = f"{data_dir}/downstream_sample_seed-{seed}.json"

    with open(path_sample_split, "r") as f:
        if stage in ["train", "validation", "test"]:
            subj_list = json.load(f)[stage]
        else:
            subj_list = json.load(f)["test_downstreams"][stage]
    return load_ukbb_dset_path(
        participant_id=subj_list,
        atlas_desc=f"atlas-{atlas[0]}_desc-{atlas[1]}",
        segment=0,  # use full time series, decimate later in make_sequence_single_subject
    )


def create_hold_out_sample(
    phenotype_path: str,
    phenotype_meta: str,
    class_balance_confounds: List[str],
    train_val_test_split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    random_state: int = 42,
) -> Dict:
    """Create experiment sample with patients in the hold out set.

    Args:
        phenotype_path (Union[Path, str]): Path to the tsv file.
            Column index 0 must be participant_id.
        phenotype_meta (Union[Path, str]): Path to the json file.
        confounds (List[str]): list of confounds to use for class
            balancing.
        hold_out_set (float, optional): proportion of the test set size
            in relation to the full sample. Defaults to 0.25.
        random_state (int, optional): random state for reproducibility.
    Returns:
        dict: dictionary with list of participant ID for training and
            hold out set, and the downstream task samples.
    """
    with open(phenotype_meta, "r") as f:
        meta = json.load(f)

    data = pd.read_csv(phenotype_path, sep="\t", index_col=0)

    diagnosis_groups = list(meta["diagnosis"]["labels"].keys())
    diagnosis_groups.remove("HC")

    n_sample = data.shape[0]

    # create a hold out set for downstream analysis including all
    # the patients
    any_patients = data[diagnosis_groups].sum(axis=1) > 0
    patients = list(data[any_patients].index)
    controls = list(data[~any_patients].index)

    n_patients = len(patients)
    n_control = n_sample - n_patients
    n_control_in_test_set = int(
        n_sample * train_val_test_split[-1] - n_patients
    )

    corrected_test_set = n_control_in_test_set / n_control
    controls_site = list(data[~any_patients]["site"])
    train_val, test = train_test_split(
        controls,
        test_size=corrected_test_set,
        random_state=random_state,
        stratify=controls_site,
    )
    test += patients

    # get controls that matches patients confounds
    data_test = data.loc[test]
    downstreams = {}
    for d in diagnosis_groups:
        select_sample = gcb.class_balance(
            classes=data_test[d].values.astype(int),
            confounds=data_test[class_balance_confounds].values.T,
            plim=0.05,
            random_seed=random_state,  # fix random seed for reproducibility
        )
        selected = data_test.index[select_sample].tolist()
        selected.sort()
        downstreams[d] = selected

    # split the training and validation set
    train, val = train_test_split(
        train_val,
        test_size=train_val_test_split[1]
        / (train_val_test_split[0] + train_val_test_split[1]),
        random_state=random_state,
    )
    del train_val
    train.sort()
    val.sort()
    test.sort()
    return {
        "train": train,
        "validation": val,
        "test": test,
        "test_downstreams": downstreams,
    }


if __name__ == "__main__":
    # test the data module
    _ = UKBBDataModule(
        connectome_file="inputs/connectomes/ukbb_libral_scrub_20240716_connectome.h5",
        phenotype_file="inputs/connectomes/ukbb_libral_scrub_20240716_phenotype.tsv",
        data_dir="inputs/data/",
        atlas=("MIST", 197),
        proportion_sample=1.0,
        timeseries_decimate=4,
        timeseries_window_stride_lag=(16, 1, 1),
        train_val_test_split=(0.6, 0.2, 0.2),
        class_balance_confounds=[
            "site",
            "sex",
            "age",
            "mean_fd_raw",
            "proportion_kept",
        ],
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        random_state=1,
    )
