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
    path_success_job = Path(
        "outputs/autoreg/full_experiment/ccn2024_scaling"
    ).glob("*/*/simple_classifiers_sex.tsv")
    path_success_job = peek(path_success_job)

    scaling_stats = pd.DataFrame()
    for p in path_success_job:
        # parse the path and get number of subjects
        n_sample = int(p.parts[-2].split("n-")[-1].split("_")[0])
        if n_sample == -1:
            n_sample = 25992
        # get random seed
        random_seed = int(p.parts[-2].split("seed-")[-1].split("_")[0])
        # load r2_val.npy get mean r2
        mean_r2_val = np.load(p.parent / "mean_r2_val.npy").tolist()
        mean_r2_tng = np.load(p.parent / "mean_r2_tng.npy").tolist()
        # get runtime from log file text
        log_file = p.parent / "full_experiment.log"
        with open(log_file, "r") as f:
            log_text = f.read()
            starttime = re.search(
                r"\[([\d\-\s:,]*)\].*Process ID", log_text
            ).group(1)
            endtime = re.search(
                r"\[([\d\-\s:,]*)\].*model trained", log_text
            ).group(1)
            starttime = pd.to_datetime(starttime)
            endtime = pd.to_datetime(endtime)
            runtime = endtime - starttime
            # convert to log scale
            runtime = runtime.total_seconds() / 60
            runtime_log = np.log10(runtime)

        # load connectome accuracy
        prediction = pd.read_csv(
            p.parent / "simple_classifiers_sex.tsv", sep="\t", index_col=0
        )
        prediction = prediction.loc[
            prediction["classifier"] == "SVM", ["feature", "score"]
        ]
        prediction = prediction.set_index("feature")
        prediction = prediction.T.reset_index(drop=True)

        df = pd.DataFrame(
            [
                n_sample,
                random_seed,
                mean_r2_val,
                mean_r2_tng,
                runtime,
                runtime_log,
            ],
            index=[
                "n_sample",
                "random_seed",
                "mean_r2_val",
                "mean_r2_tng",
                "runtime",
                "runtime_log",
            ],
        ).T
        df = pd.concat([df, prediction], axis=1)
        scaling_stats = pd.concat([scaling_stats, df], axis=0)

    # sort by n_sample
    scaling_stats = scaling_stats.sort_values(by="n_sample")
    # for each n_sample, sort by random seed
    scaling_stats = scaling_stats.groupby("n_sample").apply(
        lambda x: x.sort_values(by="random_seed")
    )
    scaling_stats = scaling_stats.reset_index(drop=True)

    scaling_stats.to_csv("outputs/ccn2024/scaling_data.csv")
    # stats[name] = scaling_stats
    # alternative data to show missing experiment
    # random seed as column and runtime as value
    scaling_overview = scaling_stats.pivot(
        index="n_sample", columns="random_seed", values="mean_r2_val"
    )

    # give a summary of the random seed and n_sample pair
    # with no runtime. this is because the experiment failed
    incomplete_n_sample = scaling_overview.isna().sum(axis=1)
    incomplete_n_sample = incomplete_n_sample[incomplete_n_sample > 0]
    # make sure all possible n_sample are included
    for n_sample in scaling_overview.index:
        if n_sample not in incomplete_n_sample.index:
            incomplete_n_sample[n_sample] = 0
    incomplete_n_sample = incomplete_n_sample.sort_index()
    missing_experiment = {}
    for n_sample in incomplete_n_sample.index:
        missing_experiment[n_sample] = scaling_overview.columns[
            scaling_overview.loc[n_sample].isna()
        ].tolist()
    # save to json
    with open(
        "outputs/ccn2024/scaling_missing_experiment.json",
        "w",
    ) as f:
        json.dump(missing_experiment, f, indent=2)

    plt.figure(figsize=(7, 4.5))
    # plot
    sns.lineplot(
        data=scaling_stats,
        x="n_sample",
        y="mean_r2_tng",
        marker="o",
        label="Traing set",
    )
    sns.lineplot(
        data=scaling_stats,
        x="n_sample",
        y="mean_r2_val",
        marker="o",
        label="Validation set",
    )
    plt.ylim(0.12, 0.19)
    plt.xlabel("Number of subject in the scaling experiment")
    plt.ylabel("R-squared")
    plt.legend()
    plt.title("R-squared of t+1 prediction")
    plt.savefig("outputs/ccn2024/scaling_r2_tng_plot.png")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    sns.lineplot(
        data=scaling_stats,
        x="n_sample",
        y="runtime_log",
        marker="o",
    )
    plt.xlabel("Number of subject in the scaling experiment")
    plt.ylabel("log10(runtime) (minutes)")
    plt.title("Runtime of training a group model")
    plt.savefig("outputs/ccn2024/scaling_runtime_plot.png")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    # plot
    features = prediction.columns.tolist()
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
        sns.lineplot(
            data=scaling_stats, x="n_sample", y=y, marker="o", label=label
        )
    plt.xlabel("Number of subject in the scaling experiment")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Sex prediction accuracy with SVM")
    plt.savefig("outputs/ccn2024/scaling_connectome.png")
    plt.close()


if __name__ == "__main__":
    main()
