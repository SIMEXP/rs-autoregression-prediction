"""
python src/create_holdout_sample.py --multirun \
    +data=ukbb ++random_state=1,2,3,5,7,10,42,435,764,9999
This is script will create all the input/labels for the full dataset.
"""
import json
import logging
import os
from pathlib import Path

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fmri_autoreg.data.load_data import get_edge_index, load_data, make_seq
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="base")
def main(params: DictConfig) -> None:
    from src.data.load_data import create_hold_out_sample

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir)
    rng = np.random.default_rng(params["random_state"])
    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    log.info(f"Random seed {params['random_state']}")

    # create hold out sample using the full dataset and save things
    sample = create_hold_out_sample(
        phenotype_path=params["data"]["phenotype_file"],
        phenotype_meta=params["data"]["phenotype_json"],
        class_balance_confounds=params["data"]["class_balance_confounds"],
        hold_out_set=params["data"]["hold_out_set"],
        random_state=params["random_state"],
    )

    data = pd.read_csv(params["data"]["phenotype_file"], sep="\t", index_col=0)

    with open(params["data"]["phenotype_json"], "r") as f:
        meta = json.load(f)

    with open(output_dir / "downstream_sample.json", "w") as f:
        json.dump(sample, f, indent=2)

    # plot the distribution of confounds of downstreams balanced samples
    log.info("Holdout sample created")
    report_dir = output_dir / "report"
    report_dir.mkdir(exist_ok=True)
    demographics = {}
    for d in sample["test_downstreams"].keys():
        d_subjects = sample["test_downstreams"][d]
        df = data.loc[d_subjects, :]
        fig, axes = plt.subplots(
            1,
            len(params["data"]["class_balance_confounds"]),
            figsize=(20, len(params["data"]["class_balance_confounds"]) + 1),
        )
        fig.suptitle(
            f"Confound balanced sample (N={len(d_subjects)}): "
            f"{meta[d]['instance']['1']['description']}"
        )
        for ax, c in zip(axes, params["data"]["class_balance_confounds"]):
            sns.histplot(x=c, data=df, hue=d, kde=True, ax=ax)
        fig.savefig(report_dir / f"{d}.png")
        demographics[d] = {
            "patient": {
                "condition": d,
                "total": df[df[d] == 1].shape[0],
                "n_female": df[df[d] == 1].shape[0]
                - df[df[d] == 1]["sex"].sum(),
                "age_mean": df[df[d] == 1]["age"].mean(),
                "age_sd": df[df[d] == 1]["age"].std(),
                "mean_fd_mean": df[df[d] == 1]["mean_fd_raw"].mean(),
                "mean_fd_sd": df[df[d] == 1]["mean_fd_raw"].std(),
                "proportion_kept_mean": df[df[d] == 1][
                    "proportion_kept"
                ].mean(),
                "proportion_kept_sd": df[df[d] == 1]["proportion_kept"].std(),
            },
            "control": {
                "condition": d,
                "total": df[df[d] == 0].shape[0],
                "n_female": df[df[d] == 0].shape[0]
                - df[df[d] == 0]["sex"].sum(),
                "age_mean": df[df[d] == 0]["age"].mean(),
                "age_sd": df[df[d] == 0]["age"].std(),
                "mean_fd_mean": df[df[d] == 0]["mean_fd_raw"].mean(),
                "mean_fd_sd": df[df[d] == 0]["mean_fd_raw"].std(),
                "proportion_kept_mean": df[df[d] == 0][
                    "proportion_kept"
                ].mean(),
                "proportion_kept_sd": df[df[d] == 0]["proportion_kept"].std(),
            },
        }
    # save the summary
    demographics_summary = pd.DataFrame()
    for d in demographics.keys():
        df = pd.DataFrame.from_dict(demographics[d], orient="index")
        df.set_index([df.index, "condition"], inplace=True)
        demographics_summary = pd.concat([demographics_summary, df])
    demographics_summary.round(decimals=2).to_csv(
        report_dir / "demographics_summary.tsv", sep="\t"
    )

    for key in sample.keys():
        if key == "test_downstreams":
            continue
        d_subjects = sample[key]
        df = data.loc[d_subjects, :]
        fig, axes = plt.subplots(
            1,
            len(params["data"]["class_balance_confounds"]),
            figsize=(20, len(params["data"]["class_balance_confounds"]) + 1),
        )
        fig.suptitle(f"{key} sample (N={len(d_subjects)})")
        for ax, c in zip(axes, params["data"]["class_balance_confounds"]):
            sns.histplot(x=c, data=df, kde=True, ax=ax)
        fig.savefig(report_dir / f"{key}.png")

    log.info("Sample report created")

    full_train_sample = [f"sub-{s}" for s in sample["train"]][:100]
    test_participant_ids = [f"sub-{s}" for s in sample["hold_out"]]
    rng.shuffle(full_train_sample)

    # pre generate labels for training samples
    prefix = f"sample_seed-{params['random_state']}"
    data_h5 = Path(output_dir) / f"{prefix}_data-train.h5"
    original_reference = Path(output_dir) / f"{prefix}_split.json"
    data_reference = {}

    # further split the training sample into training and validation

    log.info(
        f"Create dataset of {len(full_train_sample)} subjects "
        "for pretrain model. "
    )
    train_participant_ids, val_participant_ids = train_test_split(
        full_train_sample,
        test_size=params["data"]["validation_set"],
        shuffle=False,
        random_state=params["random_state"],
    )
    data_ids = (
        train_participant_ids,
        val_participant_ids,
        test_participant_ids,
    )
    # save reference to the h5 path in the original data file
    data_reference = create_reference(params, data_ids)

    # generate labels for the autoregressive model
    with h5py.File(data_h5, "a") as f:
        for n_embed in data_reference.keys():
            base = f"n_embed-{n_embed}"
            log.info(f"Creating dataset for n_embed-{n_embed}")
            f.create_group(base)
            for split in ["train", "val"]:
                cur_group = f.create_group(f"{base}/{split}")

                if split == "train":
                    # use the training set (exclude validation set)
                    # to create the connectome
                    edges = get_edge_index(
                        data_file=params["data"]["data_file"],
                        dset_paths=data_reference[n_embed]["train"],
                    )
                    f[f"n_embed-{n_embed}"]["train"].create_dataset(
                        "connectome", data=edges
                    )

                for dset in tqdm(
                    data_reference[n_embed][split],
                    desc=f"Creating {split} set",
                ):
                    data = load_data(
                        path=params["data"]["data_file"],
                        h5dset_path=dset,
                        standardize=False,
                        dtype="data",
                    )
                    x, y = make_seq(
                        data,
                        params["data"]["seq_length"],
                        params["data"]["time_stride"],
                        params["data"]["lag"],
                    )
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
                            maxshape=(
                                None,
                                n_embed,
                                params["data"]["seq_length"],
                            ),
                            chunks=(
                                x.shape[0],
                                n_embed,
                                params["data"]["seq_length"],
                            ),
                        )
                        cur_group.create_dataset(
                            name="label",
                            data=y,
                            dtype=np.float32,
                            maxshape=(None, n_embed),
                            chunks=(y.shape[0], n_embed),
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
    with open(original_reference, "a") as f:
        json.dump(data_reference, f, indent=2)


def create_reference(params, data_ids):
    data_reference = {}
    from src.data.load_data import load_ukbb_dset_path

    for n_embed in [64, 197, 444]:
        data_reference[n_embed] = {}
        for d in zip(["train", "val", "test"], data_ids):
            data_reference[n_embed][d[0]] = load_ukbb_dset_path(
                participant_id=d[1],
                atlas_desc=f"atlas-MIST_desc-{n_embed}",
                segment=params["data"]["segment"],
            )
    return data_reference


if __name__ == "__main__":
    main()
