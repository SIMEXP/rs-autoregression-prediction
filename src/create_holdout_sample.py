import json
import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hydra.utils import instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="base")
def main(params: DictConfig) -> None:
    from src.data.load_data import create_hold_out_sample

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir)

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
        fig.savefig(output_dir / f"{d}.png")
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

    demographics_summary = pd.DataFrame()
    for d in demographics.keys():
        df = pd.DataFrame.from_dict(demographics[d], orient="index")
        df.set_index([df.index, "condition"], inplace=True)
        demographics_summary = pd.concat([demographics_summary, df])
    demographics_summary.round(decimals=2).to_csv(
        output_dir / "demographics_summary.tsv", sep="\t"
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
        fig.savefig(output_dir / f"{key}.png")


if __name__ == "__main__":
    main()
