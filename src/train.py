"""
Example: Hyperparameter tuning with orion
```
python src/train.py --multirun hydra=hyperparameters
```

Example: Run scaling on different number of samples for the
model training
```
python src/train.py --multirun hydra=scaling \
  ++data.proportion_sample=1,0.5,0.25,0.1
```
"""
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
from fmri_autoreg.data.load_data import get_edge_index_threshold
from fmri_autoreg.models.train_model import train
from fmri_autoreg.tools import chebnet_argument_resolver
from omegaconf import DictConfig, OmegaConf
from seaborn import lineplot
from torchinfo import summary

LABEL_DIR = Path(__file__).parents[1] / "outputs" / "sample_for_pretraining"


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(params: DictConfig) -> None:
    """Train model using parameters dict and save results."""
    # import local library here because sumbitit and hydra being weird
    # if not interactive session of slurm, import submit it
    if (
        "SLURM_JOB_ID" in os.environ
        and os.environ["SLURM_JOB_NAME"] != "interactive"
    ):
        pid = os.getpid()
        # A logger for this file
        log = logging.getLogger(f"Process ID {pid}")
        log.info(f"Process ID {pid}")
        tng_data_h5 = (
            Path(os.environ["SLURM_TMPDIR"])
            / f"data_{os.environ['SLURM_JOB_ID']}.h5"
        )
    else:
        log = logging.getLogger(__name__)
        tng_data_h5 = list(
            (LABEL_DIR / f"seed-{params['random_state']}").glob("*train.h5")
        )[
            0
        ]  # will be shuffled after loading

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
    train_param["num_workers"] = params["num_workers"]
    train_param["torch_device"] = device
    train_param["random_state"] = params["random_state"]
    if "checkpoints" in params:
        train_param["checkpoints"] = params["checkpoints"]
    log.info(f"Working on {device}.")

    # get path data
    try:
        with h5py.File(tng_data_h5, "r") as h5file:
            connectome = h5file[f"n_embed-{train_param['n_embed']}"]["train"][
                "connectome"
            ][:]
    except OSError:
        log.error(f"File {tng_data_h5} corrupted.")
        return 1

    # get edge index
    edge_index = get_edge_index_threshold(
        connectome, train_param["edge_index_thres"]
    )
    log.info("Loaded connectome.")
    train_data = (tng_data_h5, edge_index)
    del edge_index

    with h5py.File(tng_data_h5, "r") as h5file:
        n_tng_inputs = h5file[f"n_embed-{train_param['n_embed']}"]["train"][
            "input"
        ].shape[0]
        n_tng_inputs *= train_param["proportion_sample"]

    if n_tng_inputs < train_param["batch_size"]:
        log.info(
            "Batch size is greater than the number of sequences. "
            "Setting batch size to number of sequences. "
            f"New batch size: {n_tng_inputs}. "
            f"Old batch size: {train_param['batch_size']}."
        )
        train_param["batch_size"] = n_tng_inputs
    if compute_edge_index:  # chebnet
        train_param = chebnet_argument_resolver(train_param)
    # save train_param
    with open(os.path.join(output_dir, "train_param.yaml"), "w") as f:
        OmegaConf.save(config=train_param, f=f)

    log.info("Start training.")
    (
        model,
        mean_r2_tng,
        mean_r2_val,
        losses,
        checkpoints,
    ) = train(train_param, train_data, verbose=params["verbose"])

    # save training results
    np.save(os.path.join(output_dir, "mean_r2_tng.npy"), mean_r2_tng)
    np.save(os.path.join(output_dir, "mean_r2_val.npy"), mean_r2_val)
    np.save(os.path.join(output_dir, "training_losses.npy"), losses)
    if "checkpoints" in params:
        # save a list of dictionaries as pd dataframe
        checkpoints = pd.DataFrame(checkpoints)
        checkpoints.to_csv(
            os.path.join(output_dir, "checkpoints.tsv"), sep="\t"
        )
    if params["verbose"] > 3:
        # get model info
        with open(os.path.join(output_dir, "model_info.txt"), "w") as f:
            model_stats = summary(model)
            summary_str = str(model_stats)
            f.write(summary_str)

        # get model info
        with open(
            os.path.join(output_dir, "model_info_with_input.txt"), "w"
        ) as f:
            model_stats = summary(
                model,
                input_size=(
                    train_param["batch_size"],
                    train_param["n_embed"],
                    train_param["seq_length"],
                ),
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                ],
            )
            summary_str = str(model_stats)
            f.write(summary_str)

    log.info(f"Mean r2 tng: {mean_r2_tng}")
    log.info(f"Mean r2 val: {mean_r2_val}")

    model = model.to(device)
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pk.dump(model, f)

    # visualise training loss
    training_losses = pd.DataFrame(losses)
    plt.figure()
    g = lineplot(data=training_losses)
    g.set_title(f"Training Losses (number of inputs={n_tng_inputs})")
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
