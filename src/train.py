"""
Example: Hyperparameter tuning with orion
```
python src/train.py --multirun hydra=hyperparameters
```

Example: Run scaling on different number of samples for the
model training
```
python src/train.py --multirun hydra=scaling \
  ++data.n_sample=100,200,300,-1
```
"""
import json
import logging
import os
import pickle as pk
from pathlib import Path

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fmri_autoreg.data.load_data import make_input_labels
from fmri_autoreg.models.train_model import train
from omegaconf import DictConfig
from seaborn import lineplot
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

    if (
        "SLURM_JOB_ID" in os.environ
        and os.environ["SLURM_JOB_NAME"] != "interactive"
    ):
        # import submitit
        # env = submitit.JobEnvironment()
        pid = os.getpid()
        # A logger for this file
        log = logging.getLogger(f"Process ID {pid}")
        log.info(f"Process ID {pid}")
        # use SLURM_TMPDIR for data_dir
        data_dir = Path(os.environ["SLURM_TMPDIR"]) / f"pid_{pid}"
        data_dir.mkdir()
    else:
        log = logging.getLogger(__name__)
        data_dir = Path(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")

    # organise parameters
    compute_edge_index = "Chebnet" in params["model"]["model"]
    log.info(f"Random seed {params['random_state']}")
    log.info(params["model"]["model"])
    params["model"]["nb_epochs"] = int(params["model"]["nb_epochs"])

    # flatten the parameters
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_param = {**params["model"], **params["data"]}
    train_param["torch_device"] = device
    train_param["random_state"] = params["random_state"]
    log.info(f"Working on {device}.")

    # load data path
    n_sample = params["data"]["n_sample"]

    data_split_json = params["data_split"]

    with open(data_split_json, "r") as f:
        train_subject = json.load(f)["train"]
        test_subject = json.load(f)["holdout"]

    rng.shuffle(train_subject)

    if n_sample > 0:
        train_subject = train_subject[:n_sample]

    train_subject = [f"sub-{s}" for s in train_subject]
    test_subject = [f"sub-{s}" for s in test_subject]

    train_participant_ids, val_participant_ids = train_test_split(
        train_subject,
        test_size=params["data"]["validation_set"],
        shuffle=True,
        random_state=params["random_state"],
    )

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
        participant_id=test_subject,
        atlas_desc=params["data"]["atlas_desc"],
        segment=params["data"]["segment"],
    )
    with open(Path(output_dir) / "train_test_split.json", "w") as f:
        json.dump(data_reference, f, indent=2)
    n_sample_pretrain = len(data_reference["train"]) + len(
        data_reference["val"]
    )
    log.info(
        f"Experiment on {n_sample_pretrain} subjects for pretrain model. "
    )

    tng_data_h5 = data_dir / "data_train.h5"
    val_data_h5 = data_dir / "data_val.h5"
    tng_data_h5, edge_index = make_input_labels(
        data_file=params["data"]["data_file"],
        dset_paths=data_reference["train"],
        params=train_param,
        output_file_path=tng_data_h5,
        compute_edge_index=compute_edge_index,
        log=log,
    )
    val_data_h5, _ = make_input_labels(
        data_file=params["data"]["data_file"],
        dset_paths=data_reference["val"],
        params=train_param,
        output_file_path=val_data_h5,
        compute_edge_index=False,
        log=log,
    )
    if params["verbose"] > 1:
        log.info(
            f"Training data: {convert_bytes(os.path.getsize(tng_data_h5))}"
        )
        log.info(
            f"Validation data: {convert_bytes(os.path.getsize(val_data_h5))}"
        )

    train_data = (tng_data_h5, val_data_h5, edge_index)
    del edge_index

    with h5py.File(tng_data_h5, "r") as h5file:
        n_seq = h5file["input"].shape[0]
    if n_seq < train_param["batch_size"]:
        log.info(
            "Batch size is greater than the number of sequences. "
            "Setting batch size to number of sequences. "
            f"New batch size: {n_seq}. "
            f"Old batch size: {train_param['batch_size']}."
        )
        train_param["batch_size"] = n_seq

    log.info("Start training.")
    (
        model,
        mean_r2_tng,
        mean_r2_val,
        losses,
        _,
    ) = train(train_param, train_data, verbose=params["verbose"])
    # save training results
    np.save(os.path.join(output_dir, "mean_r2_tng.npy"), mean_r2_tng)
    np.save(os.path.join(output_dir, "mean_r2_val.npy"), mean_r2_val)
    np.save(os.path.join(output_dir, "training_losses.npy"), losses)
    log.info(f"Mean r2 tng: {mean_r2_tng}")
    log.info(f"Mean r2 val: {mean_r2_val}")

    model = model.to(device)
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pk.dump(model, f)

    # visualise training loss
    training_losses = pd.DataFrame(losses)
    plt.figure()
    g = lineplot(data=training_losses)
    g.set_title(f"Training Losses (N={n_sample})")
    g.set_xlabel("Epoc")
    g.set_ylabel("Loss (MSE)")
    plt.savefig(Path(output_dir) / "training_losses.png")

    # minimise objective for hyperparameter tuning
    if mean_r2_val < 0 or mean_r2_val > 1:
        return 1  # invalid r2 score means bad fit
    else:
        return 1 - mean_r2_val


if __name__ == "__main__":
    main()
