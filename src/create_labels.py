"""
python src/create_labels.py \
    model=basic_model \
    ++data_split=path/to/downstream_sample.json
"""
import json
import logging
import os
from pathlib import Path

import h5py
import hydra
import numpy as np
from fmri_autoreg.data.load_data import make_input_labels
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def convert_bytes(num):
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:.1f} {x}"
        num /= 1024.0


log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(params: DictConfig) -> None:
    """Train model using parameters dict and save results."""
    # import local library here because sumbitit and hydra being weird
    # if not interactive session of slurm, import submit it
    from src.data.load_data import load_ukbb_dset_path

    rng = np.random.default_rng(params["random_state"])

    log = logging.getLogger(__name__)
    data_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")

    # organise parameters
    log.info(f"Random seed {params['random_state']}")
    log.info(params["model"]["model"])

    # flatten the parameters
    train_param = {**params["model"], **params["data"]}
    train_param["random_state"] = params["random_state"]

    # load data path
    proportion_sample = params["data"]["proportion_sample"]

    data_split_json = params["data_split"]

    with open(data_split_json, "r") as f:
        train_subject = json.load(f)["train"]

    with open(data_split_json, "r") as f:
        test_subject = json.load(f)["hold_out"]

    rng.shuffle(train_subject)

    n_sample = len(train_subject)
    if proportion_sample < 1:
        n_sample = int(len(train_subject) * proportion_sample)
        train_subject = train_subject[:n_sample]

    train_subject = [f"sub-{s}" for s in train_subject]

    train_participant_ids, val_participant_ids = train_test_split(
        train_subject,
        test_size=params["data"]["validation_set"],
        shuffle=True,
        random_state=params["random_state"],
    )

    test_participant_ids = [f"sub-{s}" for s in test_subject]

    data_reference = {}
    data_reference["train"] = load_ukbb_dset_path(
        participant_id=train_participant_ids,
        atlas_desc=params["data"]["atlas_desc"],
        segment=params["data"]["segment"],
    )
    data_reference["val"] = load_ukbb_dset_path(
        participant_id=val_participant_ids,
        atlas_desc=params["data"]["atlas_desc"],
        segment=params["data"]["segment"],
    )
    data_reference["test"] = load_ukbb_dset_path(
        participant_id=test_participant_ids,
        atlas_desc=params["data"]["atlas_desc"],
        segment=params["data"]["segment"],
    )
    with open(Path(output_dir) / "train_test_split.json", "w") as f:
        json.dump(data_reference, f, indent=2)
    n_sample_pretrain = len(data_reference["train"]) + len(
        data_reference["val"]
    )
    log.info(
        f"Experiment on {n_sample_pretrain} subjects "
        f"({proportion_sample * 100}% of the training sample) for pretrain model. "
    )

    prefix = f"sample_percent-{proportion_sample * 100}_seed-{params['random_state']}"
    tng_data_h5 = data_dir / f"{prefix}_data-train.h5"
    val_data_h5 = data_dir / f"{prefix}_data-val.h5"
    tng_data_h5, connectome, _ = make_input_labels(
        data_file=params["data"]["data_file"],
        dset_paths=data_reference["train"],
        params=train_param,
        output_file_path=tng_data_h5,
        compute_edges=True,
        log=log,
    )
    val_data_h5, _, _ = make_input_labels(
        data_file=params["data"]["data_file"],
        dset_paths=data_reference["val"],
        params=train_param,
        output_file_path=val_data_h5,
        compute_edges=False,
        log=log,
    )
    # save connectome for training to h5
    with h5py.File(tng_data_h5, "a") as f:
        f.create_dataset("group_connectome", data=connectome)

    log.info(f"Training data: {convert_bytes(os.path.getsize(tng_data_h5))}")
    log.info(f"Validation data: {convert_bytes(os.path.getsize(val_data_h5))}")


if __name__ == "__main__":
    main()
