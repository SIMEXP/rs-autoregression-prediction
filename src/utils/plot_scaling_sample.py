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
import yaml

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return itertools.chain([first], iterable)


# BASE_PATH = "outputs/neuroips-workshop_2024/scale-sample_bestmodel_different-num_workers"  # noqa
BASE_PATH = "outputs/neuroips-workshop_2024/scale-sample_bestmodel"


def main():
    path_success_job = Path(BASE_PATH).glob("data/**/training_losses.npy")
    path_success_job = peek(path_success_job)

    scaling_stats = pd.DataFrame()
    for p in path_success_job:
        # parse the path and get number of subjects
        log_file = p.parent / "train.log"
        with open(log_file, "r") as f:
            log_text = f.read()
        # parse the path and get number of subjects
        n_sample = int(
            re.search(r"Using ([\d]*) samples for training", log_text).group(1)
        )
        # get random seed
        random_seed = int(re.search(r"Random seed ([\d]*)", log_text).group(1))
        # load r2_val.npy get mean r2
        mean_r2_val = np.load(p.parent / "mean_r2_val.npy").tolist()
        mean_r2_tng = np.load(p.parent / "mean_r2_tng.npy").tolist()
        # get runtime from log file text
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
        # total number of parameters
        model_info_file = p.parent / "model_info_with_input.txt"
        if model_info_file.exists():
            with open(model_info_file, "r") as f:
                model_info = f.read()
            total_parameters = int(
                re.search(r"Total params: ([\d,]*)", model_info)
                .group(1)
                .replace(",", "")
            )
            total_mult = float(
                re.search(
                    r"Total mult-adds \(M\): ([\d.]*)", model_info
                ).group(1)
            )
            total_size = float(
                re.search(
                    r"Estimated Total Size \(MB\): ([\d.]*)", model_info
                ).group(1)
            )
        else:
            total_parameters = np.nan
            total_mult = np.nan
            total_size = np.nan
        # # load connectome accuracy
        # prediction = pd.read_csv(
        #     p.parent / "simple_classifiers_sex.tsv", sep="\t", index_col=0  # noqa
        # )
        # prediction = prediction.loc[
        #     prediction["classifier"] == "SVM", ["feature", "score"]
        # ]
        # prediction = prediction.set_index("feature")
        # prediction = prediction.T.reset_index(drop=True)

        df = pd.DataFrame(
            [
                n_sample,
                random_seed,
                mean_r2_val,
                mean_r2_tng,
                runtime,
                runtime_log,
                total_parameters,
                total_mult,
                total_size,
            ],
            index=[
                "n_sample_train",
                "random_seed",
                "mean_r2_val",
                "mean_r2_tng",
                "runtime",
                "runtime_log",
                "total_parameters",
                "total_mult",
                "total_size",
            ],
        ).T
        # df = pd.concat([df, prediction], axis=1)
        scaling_stats = pd.concat([scaling_stats, df], axis=0)

    # sort by n_sample
    scaling_stats = scaling_stats.sort_values(by="n_sample_train")
    # for each n_sample, sort by random seed
    scaling_stats = scaling_stats.groupby("n_sample_train").apply(
        lambda x: x.sort_values(by="random_seed")
    )
    scaling_stats = scaling_stats.reset_index(drop=True)
    scaling_stats["percent_sample"] = scaling_stats["n_sample_train"] / 2328583
    scaling_stats["percent_sample"] = (
        scaling_stats["percent_sample"].round(3) * 100
    )

    scaling_stats.to_csv(Path(BASE_PATH) / "reports/scaling_data.tsv", "\t")

    # alternative data to show missing experiment
    # random seed as column and runtime as value
    scaling_overview = scaling_stats.pivot(
        index="percent_sample", columns="random_seed", values="mean_r2_val"
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
        Path(BASE_PATH) / "reports/scaling_missing_experiment.json",
        "w",
    ) as f:
        json.dump(missing_experiment, f, indent=2)

    plt.figure(figsize=(5, 5))
    # plot
    sns.lineplot(
        data=scaling_stats,
        x="percent_sample",
        y="mean_r2_tng",
        marker="o",
        label="Traing set",
    )
    sns.lineplot(
        data=scaling_stats,
        x="percent_sample",
        y="mean_r2_val",
        marker="o",
        label="Validation set",
    )
    plt.ylim(0.145, 0.185)
    plt.xticks([0, 5, 10, 25, 50, 100])
    plt.xlabel("Percentage of training sample")
    plt.ylabel("R-squared")
    plt.legend()
    plt.title("R-squared of t+1 prediction")
    plt.tight_layout()
    plt.savefig(Path(BASE_PATH) / "reports/scaling_r2_tng_plot.png")
    plt.close()

    plt.figure(figsize=(5, 5))
    sns.lineplot(
        data=scaling_stats,
        x="percent_sample",
        y="runtime_log",
        marker="o",
    )
    plt.xticks([0, 5, 10, 25, 50, 100])
    plt.xlabel("Percentage of training sample")
    plt.ylabel("log10(runtime) (minutes)")
    plt.title("Runtime of training")
    plt.tight_layout()
    plt.savefig(Path(BASE_PATH) / "reports/scaling_runtime_plot.png")
    plt.close()


if __name__ == "__main__":
    main()
