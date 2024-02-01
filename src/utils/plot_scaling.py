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


PATH_PATTERN_SUCCESS_JOB = {
    'FK="8,3,8,3,8,3";M="1"': "multiruns/*/++experiment.scaling.n_sample=*,++experiment.scaling.segment=?,++random_state=*/r2_test.npy",
    'FK="32,6,32,6,32,6,16,6,16,6";M="32,16,8,1"': 'multiruns/*/++experiment.scaling.n_sample=*,++experiment.scaling.segment=1,++model.FK="32,6,32,6,32,6,16,6,16,6",++model.M="32,16,8,1",++model.dropout=0.1,++random_state=*/r2_test.npy',
    'FK="128,32,128,32,128,32,128,32,128,32,64,16,64,16";M="128,64,32,16,8,1"': 'multiruns/*/++experiment.scaling.n_sample=*,++experiment.scaling.segment=1,++model.FK="128,32,128,32,128,32,128,32,128,32,64,16,64,16",++model.M="128,64,32,16,8,1",++model.dropout=0.1,++random_state=*/r2_test.npy',
}


def main():
    stats = {}
    for name, pattern in PATH_PATTERN_SUCCESS_JOB.items():
        path_success_job = Path("outputs/scaling").glob(pattern)
        path_success_job = peek(path_success_job)

        # if no path found, skip to next item
        if path_success_job is None:
            continue

        scaling_stats = pd.DataFrame()
        for p in path_success_job:
            # parse the path and get number of subjects
            n_sample = int(p.parts[-2].split("n_sample=")[-1].split(",")[0])
            if n_sample == -1:
                n_sample = 25992
            # get random seed
            random_seed = int(
                p.parts[-2].split("random_state=")[-1].split(",")[0]
            )
            # load r2_val.npy get mean r2
            r2_test = np.load(p)
            mean_r2_test = r2_test.mean()
            r2_val = np.load(p.parent / "r2_val.npy")
            mean_r2_val = r2_val.mean()
            r2_tng = np.load(p.parent / "r2_tng.npy")
            mean_r2_tng = r2_tng.mean()
            # get runtime from log file text
            log_file = p.parent / "scaling.log"
            with open(log_file, "r") as f:
                log_text = f.read()
                starttime = re.search(
                    r"\[([\d\-\s:,]*)\].*Process ID", log_text
                ).group(1)
                endtime = re.search(
                    r"\[([\d\-\s:,]*)\].*completed", log_text
                ).group(1)
                starttime = pd.to_datetime(starttime)
                endtime = pd.to_datetime(endtime)
                runtime = endtime - starttime
                # convert to log scale
                runtime = runtime.total_seconds() / 60
                runtime_log = np.log10(runtime)

            df = pd.DataFrame(
                [
                    n_sample,
                    random_seed,
                    mean_r2_test,
                    mean_r2_val,
                    mean_r2_tng,
                    runtime,
                    runtime_log,
                ],
                index=[
                    "n_sample",
                    "random_seed",
                    "mean_r2_test",
                    "mean_r2_val",
                    "mean_r2_tng",
                    "runtime",
                    "runtime_log",
                ],
            ).T
            scaling_stats = pd.concat([scaling_stats, df], axis=0)

        # sort by n_sample
        scaling_stats = scaling_stats.sort_values(by="n_sample")
        # for each n_sample, sort by random seed
        scaling_stats = scaling_stats.groupby("n_sample").apply(
            lambda x: x.sort_values(by="random_seed")
        )
        scaling_stats = scaling_stats.reset_index(drop=True)

        scaling_stats.to_csv(f"outputs/scaling/model-{name}_scaling_data.csv")
        stats[name] = scaling_stats
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
            f"outputs/scaling/model-{name}_scaling_missing_experiment.json",
            "w",
        ) as f:
            json.dump(missing_experiment, f, indent=2)

    plt.figure(figsize=(7, 5))
    for key, df in stats.items():
        # plot
        sns.lineplot(
            data=df,
            x="n_sample",
            y="mean_r2_tng",
            marker="o",
            label=key.replace(";", "\n"),
        )
    plt.ylabel("R^2")
    plt.legend()
    plt.title("R^2 of the Training set")
    plt.savefig("outputs/scaling/scaling_r2_tng_plot.png")
    plt.close()
    plt.figure(figsize=(7, 5))
    for key, df in stats.items():
        # plot
        sns.lineplot(
            data=df,
            x="n_sample",
            y="mean_r2_val",
            marker="o",
            label=key.replace(";", "\n"),
        )
    plt.ylabel("R^2")
    plt.legend()
    plt.title("R^2 of the Validation set")
    plt.savefig("outputs/scaling/scaling_r2_val_plot.png")
    plt.close()
    plt.figure(figsize=(7, 5))
    for key, df in stats.items():
        # plot
        sns.lineplot(
            data=df,
            x="n_sample",
            y="mean_r2_test",
            marker="o",
            label=key.replace(";", "\n"),
        )
    plt.ylabel("R^2")
    plt.legend()
    plt.title("R^2 of the Test set")
    plt.savefig("outputs/scaling/scaling_r2_test_plot.png")
    plt.close()
    plt.figure(figsize=(7, 5))
    for key, df in stats.items():
        sns.lineplot(
            data=df,
            x="n_sample",
            y="runtime_log",
            marker="o",
            label=key.replace(";", "\n"),
        )
    plt.ylabel("log10(runtime) (minutes)")
    plt.title("Runtime of training a group model")
    plt.legend()
    plt.savefig("outputs/scaling/scaling_runtime_plot.png")
    plt.close()


if __name__ == "__main__":
    main()
