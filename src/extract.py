"""
Extract features from the model.
If model was trained on gpu, this script should
be run on a machine with a gpu.
```
python src/extract.py --multirun \
    model_path=outputs/ccn2024/model/model.pkl
```
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import h5py
import hydra
import torch
from fmri_autoreg.models.predict_model import predict_horizon
from fmri_autoreg.tools import load_model
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="extract")
def main(params: DictConfig) -> None:
    """Train model using parameters dict and save results."""

    from src.model.extract_features import (
        extract_convlayers,
        pooling_convlayers,
    )

    model_path = Path(params["model_path"])
    model_config = OmegaConf.load(model_path.parent / ".hydra/config.yaml")
    params = OmegaConf.merge(model_config, params)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Working on {device}.")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir)
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    if isinstance(params["horizon"], int):
        horizons = [params["horizon"]]
    else:
        horizons = OmegaConf.to_object(params["horizon"])
    log.info(f"predicting horizon: {horizons}")

    # load test set subject path from the training
    with open(model_path.parent / "train_test_split.json", "r") as f:
        subj = json.load(f)

    subj_list = subj["test"]

    # save test data path to a text file for easy future reference
    with open(output_dir / "test_set_connectome.txt", "w") as f:
        for item in subj_list:
            f.write("%s\n" % item)

    log.info("Load model")
    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()

    for horizon in horizons:
        output_horizon_path = (
            Path(output_dir) / f"feature_horizon-{horizon}.h5"
        )
        with h5py.File(output_horizon_path, "a") as f:
            f.attrs["complied_date"] = str(datetime.today())
            f.attrs["based_on_model"] = str(model_path)
            f.attrs["horizon"] = horizon

        log.info(f"Predicting t+{horizon} of each subject")
        for h5_dset_path in tqdm(subj_list):
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

    output_conv_path = Path(output_dir) / "feature_convlayers.h5"
    # save the model parameters in the h5 files
    with h5py.File(output_conv_path, "a") as f:
        f.attrs["complied_date"] = str(datetime.today())
        f.attrs["based_on_model"] = str(model_path)

    log.info("extract convo layers")
    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()
    for h5_dset_path in tqdm(subj_list):
        convlayers = extract_convlayers(
            data_file=params["data"]["data_file"],
            h5_dset_path=h5_dset_path,
            model=model,
            seq_length=params["model"]["seq_length"],
            time_stride=params["model"]["time_stride"],
            lag=params["model"]["lag"],
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


if __name__ == "__main__":
    main()
