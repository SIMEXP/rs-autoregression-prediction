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
