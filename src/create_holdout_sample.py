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
from src.data.load_data import create_hold_out_sample

class_balance_confounds = [
    "site",
    "sex",
    "age",
    "mean_fd_raw",
    "proportion_kept",
]
hold_out_set = 0.2

phenotype_path = "inputs/connectomes/ukbb_libral_scrub_20240716_phenotype.tsv"
phenotype_meta = "inputs/connectomes/ukbb_libral_scrub_20240716_phenotype.json"
output_downstream_sample = (
    "inputs/connectomes/ukbb_libral_scrub_20240716_downstream_sample.json"
)
output_reports = "outputs/downstream_sample_reports_20240716"

sample = create_hold_out_sample(
    phenotype_path=phenotype_path,
    phenotype_meta=phenotype_meta,
    class_balance_confounds=class_balance_confounds,
    hold_out_set=hold_out_set,
    random_state=42,
)

data = pd.read_csv(phenotype_path, sep="\t", index_col=0)

with open(phenotype_meta, "r") as f:
    meta = json.load(f)

with open(output_downstream_sample, "w") as f:
    json.dump(sample, f, indent=2)

# plot the distribution of confounds of downstreams balanced samples
demographics = {}
for d in sample["test_downstreams"].keys():
    d_subjects = sample["test_downstreams"][d]
    df = data.loc[d_subjects, :]
    fig, axes = plt.subplots(
        1,
        len(class_balance_confounds),
        figsize=(20, len(class_balance_confounds) + 1),
    )
    fig.suptitle(
        f"Confound balanced sample (N={len(d_subjects)}): "
        f"{meta[d]['instance']['1']['description']}"
    )
    for ax, c in zip(axes, class_balance_confounds):
        sns.histplot(x=c, data=df, hue=d, kde=True, ax=ax)
    fig.savefig(output_reports + f"/{d}.png")
    demographics[d] = {
        "patient": {
            "condition": d,
            "total": df[df[d] == 1].shape[0],
            "n_female": df[df[d] == 1].shape[0] - df[df[d] == 1]["sex"].sum(),
            "age_mean": df[df[d] == 1]["age"].mean(),
            "age_sd": df[df[d] == 1]["age"].std(),
            "mean_fd_mean": df[df[d] == 1]["mean_fd_raw"].mean(),
            "mean_fd_sd": df[df[d] == 1]["mean_fd_raw"].std(),
            "proportion_kept_mean": df[df[d] == 1]["proportion_kept"].mean(),
            "proportion_kept_sd": df[df[d] == 1]["proportion_kept"].std(),
        },
        "control": {
            "condition": d,
            "total": df[df[d] == 0].shape[0],
            "n_female": df[df[d] == 0].shape[0] - df[df[d] == 0]["sex"].sum(),
            "age_mean": df[df[d] == 0]["age"].mean(),
            "age_sd": df[df[d] == 0]["age"].std(),
            "mean_fd_mean": df[df[d] == 0]["mean_fd_raw"].mean(),
            "mean_fd_sd": df[df[d] == 0]["mean_fd_raw"].std(),
            "proportion_kept_mean": df[df[d] == 0]["proportion_kept"].mean(),
            "proportion_kept_sd": df[df[d] == 0]["proportion_kept"].std(),
        },
    }

demographics_summary = pd.DataFrame()
for d in demographics.keys():
    df = pd.DataFrame.from_dict(demographics[d], orient="index")
    df.set_index([df.index, "condition"], inplace=True)
    demographics_summary = pd.concat([demographics_summary, df])
demographics_summary.round(decimals=2).to_csv(
    output_reports + "/demographics_summary.tsv", sep="\t"
)

for key in sample.keys():
    if key == "test_downstreams":
        continue
    d_subjects = sample[key]
    df = data.loc[d_subjects, :]
    fig, axes = plt.subplots(
        1,
        len(class_balance_confounds),
        figsize=(20, len(class_balance_confounds) + 1),
    )
    fig.suptitle(f"{key} sample (N={len(d_subjects)})")
    for ax, c in zip(axes, class_balance_confounds):
        sns.histplot(x=c, data=df, kde=True, ax=ax)
    fig.savefig(output_reports + f"/{key}.png")
