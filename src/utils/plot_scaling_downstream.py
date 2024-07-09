"""
look through the `outputs/` directory, find instance of completed
training, and get the number of subjects used, mean R2 of test set,
plot the number of subjects (y-axis) against R2 (x axis)
"""
import itertools
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return itertools.chain([first], iterable)


def main():
    path_success_job = Path("outputs/autoreg/predict/downstream").glob(
        "**/simple_classifiers_sex.tsv"
    )
    # path_success_job = peek(path_success_job)

    scaling_stats = pd.DataFrame()
    for p in path_success_job:
        log_file = p.parent / "predict.log"
        with open(log_file, "r") as f:
            log_text = f.read()
        # parse the path and get number of subjects
        n_sample = re.search(
            r"Subjects with phenotype data: ([\d]*)", log_text
        ).group(1)
        n_sample = int(n_sample)
        percent_sample = re.search(
            r"([\d]*)% of the full sample", log_text
        ).group(1)
        # get random seed
        random_seed = re.search(r"'random\_state': ([\d]+)", log_text).group(1)
        # load connectome accuracy
        prediction = pd.read_csv(p, sep="\t", index_col=0)
        prediction["percent_sample"] = int(percent_sample)
        prediction["n_sample"] = n_sample
        prediction["random_seed"] = random_seed

        scaling_stats = pd.concat([scaling_stats, prediction], axis=0)

    # sort by n_sample
    scaling_stats = scaling_stats.sort_values(by="n_sample")
    # for each n_sample, sort by random seed
    scaling_stats = scaling_stats.groupby("n_sample").apply(
        lambda x: x.sort_values(by="random_seed")
    )
    scaling_stats = scaling_stats.reset_index(drop=True)

    scaling_stats.to_csv(
        "outputs/autoreg/predict/downstream/downstream_scaling_data.csv"
    )

    mask = scaling_stats["classifier"] == "SVM"
    plt.figure(figsize=(7, 4.5))
    # plot
    features = prediction["feature"].unique().tolist()
    for y, label in zip(
        features,
        [
            "connectomes",
            "average pooling",
            "standard deviation pooling",
            "max pooling",
            "1D convolution",
            "average R-squared",
            "R-squared map",
        ],
    ):
        feat_mask = scaling_stats["feature"] == y
        cur_mask = mask & feat_mask
        sns.lineplot(
            data=scaling_stats[cur_mask],
            x="percent_sample",
            y="score",
            marker="o",
            label=label,
        )
    plt.xlabel("Percent of subject in the downstream prediction.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(
        "Sex prediction accuracy with SVM with saturated pretrained model."
    )
    plt.savefig("outputs/autoreg/predict/downstream/downstream_scaling.png")
    plt.close()


if __name__ == "__main__":
    main()
