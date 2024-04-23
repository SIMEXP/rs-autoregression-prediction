import csv
import json
import logging
import os
import pickle as pk
import re
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from fmri_autoreg.data.load_data import make_input_labels
from fmri_autoreg.models.predict_model import predict_horizon, predict_model
from fmri_autoreg.models.train_model import train
from fmri_autoreg.tools import load_model
from hydra.utils import instantiate
from omegaconf import DictConfig
from seaborn import lineplot
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

baseline_details = {
    "connectome": {
        "data_file": None,
        "data_file_pattern": None,
        "plot_label": "Connectome",
    },
    "conv_avg": {
        "data_file": None,
        "data_file_pattern": "average",
        "plot_label": "Conv layers \n avg pooling",
    },
    "conv_std": {
        "data_file": None,
        "data_file_pattern": "std",
        "plot_label": "Conv layers \n std pooling",
    },
    "conv_max": {
        "data_file": None,
        "data_file_pattern": "max",
        "plot_label": "Conv layers \n max pooling",
    },
    "conv_conv1d": {
        "data_file": None,
        "data_file_pattern": "1dconv",
        "plot_label": "Conv layers \n 1D convolution",
    },
    "avgr2": {
        "data_file": None,
        "data_file_pattern": "r2map",
        "plot_label": "t+1\n average R2",
    },
    "r2map": {
        "data_file": None,
        "data_file_pattern": "r2map",
        "plot_label": "t+1\nR2 map",
    },
}


def convert_bytes(num):
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:.1f} {x}"
        num /= 1024.0


@hydra.main(
    version_base="1.3", config_path="../config", config_name="full_experiment"
)
def main(params: DictConfig) -> None:
    """Train model using parameters dict and save results."""
    # import local library here because sumbitit and hydra being weird
    # if not interactive session of slurm, import submit it

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
    with open(params["data_split"], "r") as f:
        data_split = json.load(f)
    data_reference = {}
    data_reference["test"] = data_split["test"]
    training_candidate = data_split["train"] + data_split["val"]
    n_sample = params["data"]["split"]["n_sample"]
    rs = ShuffleSplit(
        n_splits=1, test_size=0.25, random_state=params["random_state"]
    )
    if n_sample != -1:
        proportion = n_sample / len(training_candidate)
        sample_select = ShuffleSplit(
            n_splits=1,
            train_size=proportion,
            random_state=params["random_state"],
        )
        sample_index = next(sample_select.split(training_candidate))
        pretraining_set = [training_candidate[i] for i in sample_index]
    else:
        pretraining_set = training_candidate.copy()
    train_index, val_index = next(rs.split(pretraining_set))
    data_reference["train"] = [pretraining_set[i] for i in train_index]
    data_reference["val"] = [pretraining_set[i] for i in val_index]

    log.info(f"Pretrain model on {len(pretraining_set)} subjects.")

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
    model_path = os.path.join(output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pk.dump(model, f)

    # visualise training loss
    training_losses = pd.DataFrame(losses)
    plt.figure()
    g = lineplot(data=training_losses)
    g.set_title(f"Training Losses (N={n_sample})")
    g.set_xlabel("Epoc")
    g.set_ylabel("Loss (MSE)")
    plt.savefig(Path(output_dir) / "training_losses.png")

    log.info("Start extracting features.")
    model = model.eval()
    horizon = 1
    output_horizon_path = Path(output_dir) / f"feature_horizon-{horizon}.h5"
    with h5py.File(output_horizon_path, "a") as f:
        f.attrs["complied_date"] = str(datetime.today())
        f.attrs["based_on_model"] = str(model_path)
        f.attrs["horizon"] = horizon

    log.info(f"Predicting t+{horizon} of each subject")
    for h5_dset_path in data_reference["test"]:
        # get the prediction of t+1
        r2, Z, Y = predict_horizon(
            model=model,
            seq_length=params["model"]["seq_length"],
            horizon=horizon,
            data_file=params["data"]["data_file"],
            dset_path=h5_dset_path,
            batch_size=params["model"]["batch_size"],
            stride=params["model"]["time_stride"],
            standardize=False,  # the ts is already standardized
        )
        # save the original output to a h5 file
        with h5py.File(output_horizon_path, "a") as f:
            for value, key in zip([r2, Z, Y], ["r2map", "Z", "Y"]):
                new_ds_path = h5_dset_path.replace("timeseries", key)
                f[new_ds_path] = value

    log.info("Extract convolved layers.")
    output_conv_path = Path(output_dir) / "feature_convlayers.h5"
    # save the model parameters in the h5 files
    with h5py.File(output_conv_path, "a") as f:
        f.attrs["complied_date"] = str(datetime.today())
        f.attrs["based_on_model"] = str(model_path)

    log.info("extract convo layers")
    from src.model.extract_features import (
        extract_convlayers,
        pooling_convlayers,
    )

    model = load_model(model_path)
    thres = params["model"]["edge_index_thres"] if compute_edge_index else None
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()
    for h5_dset_path in data_reference["test"]:
        convlayers = extract_convlayers(
            data_file=params["data"]["data_file"],
            h5_dset_path=h5_dset_path,
            model=model,
            seq_length=params["model"]["seq_length"],
            time_stride=params["model"]["time_stride"],
            lag=params["model"]["lag"],
            compute_edge_index=compute_edge_index,
            thres=thres,
        )
        # save the original output to a h5 file
        with h5py.File(output_conv_path, "a") as f:
            new_ds_path = h5_dset_path.replace("timeseries", "convlayers")
            f[new_ds_path] = convlayers.numpy()
        convlayers_F = [
            int(F)
            for i, F in enumerate(params["model"]["FK"].split(","))
            if i % 2 == 0
        ]
        # get the pooling features of the assigned layer
        for method in ["average", "max", "std", "1dconv"]:
            features = pooling_convlayers(
                convlayers=convlayers,
                pooling_methods=method,
                pooling_target="parcel",
                layer_index=params["convlayer_index"],
                layer_structure=convlayers_F,
            )
            # save the original output to a h5 file
            with h5py.File(output_conv_path, "a") as f:
                new_ds_path = h5_dset_path.replace("timeseries", method)
                f[new_ds_path] = features

    # save the original output to a h5 file
    with h5py.File(output_conv_path, "a") as f:
        f.attrs["convolution_layers_F"] = convlayers_F

    log.info("Start predicting.")
    from src.data.load_data import get_model_data, load_h5_data_path

    for key in baseline_details:
        if "r2" in key:
            baseline_details[key]["data_file"] = output_horizon_path
        elif "conv" in key:
            baseline_details[key]["data_file"] = output_conv_path
        elif "connectome" in key:
            baseline_details[key]["data_file"] = params["data"]["data_file"]
            baseline_details[key]["data_file_pattern"] = data_reference["test"]
        else:
            pass
    log.info("Predicting sex.")
    # four baseline models for sex
    svm = LinearSVC(C=100, penalty="l2", max_iter=1000000, random_state=42)
    lr = LogisticRegression(
        penalty="l2", max_iter=100000, random_state=42, n_jobs=-1
    )
    rr = RidgeClassifier(random_state=42, max_iter=100000)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        max_iter=100000,
        random_state=42,
    )
    clf_names = ["SVM", "LogisticR", "Ridge", "MLP"]
    baselines_df = {
        "feature": [],
        "score": [],
        "classifier": [],
    }
    phenotype_file = Path(params["phenotype_file"])
    for measure in baseline_details:
        log.info(f"Start training {measure}")
        log.info(f"Load data {baseline_details[measure]['data_file']}")
        if measure == "connectome":
            dset_path = baseline_details[measure]["data_file_pattern"]
        else:
            dset_path = load_h5_data_path(
                baseline_details[measure]["data_file"],
                baseline_details[measure]["data_file_pattern"],
                shuffle=True,
                random_state=params["random_state"],
            )
        log.info(f"found {len(dset_path)} subjects with {measure} data.")

        dataset = get_model_data(
            baseline_details[measure]["data_file"],
            dset_path=dset_path,
            phenotype_file=phenotype_file,
            measure=measure,
            label="sex",
            log=log,
        )
        log.info("Start training...")
        tng, tst = next(
            StratifiedKFold(n_splits=5, shuffle=True).split(
                dataset["data"], dataset["label"]
            )
        )  # only one fold
        for clf_name, clf in zip(clf_names, [svm, lr, rr, mlp]):
            clf.fit(dataset["data"][tng], dataset["label"][tng])
            score = clf.score(dataset["data"][tst], dataset["label"][tst])
            log.info(f"{measure} - {clf_name} score: {score:.3f}")
            baselines_df["feature"].append(measure)
            baselines_df["score"].append(score)
            baselines_df["classifier"].append(clf_name)

    # save the results
    # json for safe keeping
    with open(
        Path(output_dir) / "simple_classifiers_sex.json",
        "w",
    ) as f:
        json.dump(baselines_df, f, indent=4)

    baselines_df = pd.DataFrame(baselines_df)
    baselines_df.to_csv(
        Path(output_dir) / "simple_classifiers_sex.tsv",
        sep="\t",
    )

    log.info("End of experiment.")


if __name__ == "__main__":
    main()
