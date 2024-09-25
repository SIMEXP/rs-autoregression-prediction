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


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return itertools.chain([first], iterable)


def main():
    path_success_job = Path(
        "outputs/neuroips-workshop_2024/scale-by-architecture"
    ).glob("scale-*/**/training_losses.npy")
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

        # read trian_param.json (which is an ymal...)
        if (p.parent / "train_param.json").exists():
            with open(p.parent / "train_param.json") as f:
                train_param = yaml.safe_load(f)
        else:
            with open(p.parent / "train_param.ymal") as f:
                train_param = yaml.safe_load(f)

        train_param["M"] = int(train_param["M"].split(",")[0])
        # total number of parameters
        model_info_file = p.parent / "model_info_with_input.txt"
        with open(model_info_file, "r") as f:
            model_info = f.read()

        total_parameters = int(
            re.search(r"Total params: ([\d,]*)", model_info)
            .group(1)
            .replace(",", "")
        )
        total_mult = float(
            re.search(r"Total mult-adds \(M\): ([\d.]*)", model_info).group(1)
        )
        total_size = float(
            re.search(
                r"Estimated Total Size \(MB\): ([\d.]*)", model_info
            ).group(1)
        )

        # # load connectome accuracy
        # prediction = pd.read_csv(
        #     p.parent / "simple_classifiers_sex.tsv", sep="\t", index_col=0
        # )
        # prediction = prediction.loc[
        #     prediction["classifier"] == "SVM", ["feature", "score"]
        # ]
        # prediction = prediction.set_index("feature")
        # prediction = prediction.T.reset_index(drop=True)

        df = pd.DataFrame(
            [
                n_sample,
                train_param["GCL"],
                train_param["F"],
                train_param["K"],
                train_param["FCL"],
                train_param["M"],
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
                "GCL",
                "F",
                "K",
                "FCL",
                "M",
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

    scaling_stats.to_csv(
        "outputs/neuroips-workshop_2024/scale-by-architecture/reports/scaling_data.tsv",
        "\t",
    )

    mask_compare_FCL = (
        (scaling_stats["GCL"] == 3)
        & (scaling_stats["F"] == 8)
        & (scaling_stats["K"] == 3)
    )

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    plot_compare_FCL = sns.heatmap(
        scaling_stats[mask_compare_FCL].pivot_table(
            index="M", columns="FCL", values="mean_r2_val"
        ),
        cmap="coolwarm",
        square=True,
        linewidth=0.5,
        vmax=0.185,
        vmin=0.16,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Mean R2 of validation set"},
        ax=ax1,
    )
    plot_compare_FCL.set_title(
        "Testing different parameters of MLP\nGCN architecture fixed; batch size ~8k"
    )
    plot_compare_FCL.set_xlabel("Number of fully connected layer")
    plot_compare_FCL.set_ylabel("Number of neurons per layer")
    # plot_compare_FCL.figure.savefig("outputs/neuroips-workshop_2024/scale-by-architecture/reports/compare_FCL.png")
    # plt.close()

    mask_compare_GCL = (scaling_stats["M"] == 8) & (scaling_stats["FCL"] == 1)

    # 3d scatter plot of F, GCL, K
    # fig = plt.figure()
    ax = fig.add_subplot(122, projection="3d")
    im = ax.scatter(
        scaling_stats[mask_compare_GCL]["F"],
        scaling_stats[mask_compare_GCL]["K"],
        scaling_stats[mask_compare_GCL]["GCL"],
        c=scaling_stats[mask_compare_GCL]["mean_r2_val"],
        cmap="coolwarm",
        s=100,
        vmin=0.16,
        vmax=0.185,
    )
    ax.set_xlabel("F")
    ax.set_xticks([8, 16, 32])
    ax.set_ylabel("K")
    ax.set_yticks([3, 5, 10])
    ax.set_zlabel("Number of layers")
    ax.set_zticks([3, 6, 9, 12])
    ax.set_title(
        "Testing different parameters of chebnet\nMLP architecture fixed; batch size ~8k"
    )
    # fig.colorbar(im, ax=ax, label="Mean R2 of validation set")
    fig.savefig(
        "outputs/neuroips-workshop_2024/scale-by-architecture/reports/compare_F-GCL-K.png"
    )
    plt.close()

    for g in [3, 6, 9, 12]:
        cur_df = mask_compare_GCL & (scaling_stats["GCL"] == g)
        plot_compare_FCL = sns.heatmap(
            scaling_stats[cur_df].pivot_table(
                index="F", columns="K", values="mean_r2_val"
            ),
            square=True,
            linewidth=0.5,
            vmax=0.185,
            vmin=0.16,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
        )
        plot_compare_FCL.set_title("Mean R2 of validation set")
        plot_compare_FCL.set_xlabel("K")
        plot_compare_FCL.set_ylabel("F")
        plot_compare_FCL.figure.savefig(
            f"outputs/neuroips-workshop_2024/scale-by-architecture/reports/compare_GCL-{g}.png"
        )
        plt.close()

    for f in [8, 16, 32]:
        cur_df = mask_compare_GCL & (scaling_stats["F"] == f)
        plot_compare_FCL = sns.heatmap(
            scaling_stats[cur_df].pivot_table(
                index="GCL", columns="K", values="mean_r2_val"
            ),
            square=True,
            linewidth=0.5,
            vmax=0.185,
            vmin=0.16,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
        )
        plot_compare_FCL.set_title("Mean R2 of validation set")
        plot_compare_FCL.set_xlabel("K")
        plot_compare_FCL.set_ylabel("Number of convolution layer")
        plot_compare_FCL.figure.savefig(
            f"outputs/neuroips-workshop_2024/scale-by-architecture/reports/compare_F-{f}.png"
        )
        plt.close()

    for k in [3, 5, 10]:
        cur_df = mask_compare_GCL & (scaling_stats["K"] == k)
        plot_compare_FCL = sns.heatmap(
            scaling_stats[cur_df].pivot_table(
                index="GCL", columns="F", values="mean_r2_val"
            ),
            square=True,
            linewidth=0.5,
            vmax=0.185,
            vmin=0.16,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
        )
        plot_compare_FCL.set_title("Mean R2 of validation set")
        plot_compare_FCL.set_xlabel("F")
        plot_compare_FCL.set_ylabel("Number of convolution layer")
        plot_compare_FCL.figure.savefig(
            f"outputs/neuroips-workshop_2024/scale-by-architecture/reports/compare_K-{k}.png"
        )
        plt.close()

    # # stats[name] = scaling_stats
    # # alternative data to show missing experiment
    # # random seed as column and runtime as value
    # scaling_overview = scaling_stats.pivot(
    #     index="n_sample", columns="random_seed", values="mean_r2_val"
    # )

    # # give a summary of the random seed and n_sample pair
    # # with no runtime. this is because the experiment failed
    # incomplete_n_sample = scaling_overview.isna().sum(axis=1)
    # incomplete_n_sample = incomplete_n_sample[incomplete_n_sample > 0]
    # # make sure all possible n_sample are included
    # for n_sample in scaling_overview.index:
    #     if n_sample not in incomplete_n_sample.index:
    #         incomplete_n_sample[n_sample] = 0
    # incomplete_n_sample = incomplete_n_sample.sort_index()
    # missing_experiment = {}
    # for n_sample in incomplete_n_sample.index:
    #     missing_experiment[n_sample] = scaling_overview.columns[
    #         scaling_overview.loc[n_sample].isna()
    #     ].tolist()
    # # save to json
    # with open(
    #     "outputs/ccn2024/scaling_missing_experiment.json",
    #     "w",
    # ) as f:
    #     json.dump(missing_experiment, f, indent=2)

    # plt.figure(figsize=(7, 4.5))
    # # plot
    # sns.lineplot(
    #     data=scaling_stats,
    #     x="n_sample_train",
    #     y="mean_r2_tng",
    #     marker="o",
    #     label="Traing set",
    # )
    # sns.lineplot(
    #     data=scaling_stats,
    #     x="n_sample_train",
    #     y="mean_r2_val",
    #     marker="o",
    #     label="Validation set",
    # )
    # plt.ylim(0.10, 0.19)
    # plt.xlabel("Number of subject in model training")
    # plt.ylabel("R-squared")
    # plt.legend()
    # plt.title("R-squared of t+1 prediction")
    # plt.savefig("outputs/ccn2024/scaling_r2_tng_plot.png")
    # plt.close()

    # plt.figure(figsize=(7, 4.5))
    # sns.lineplot(
    #     data=scaling_stats,
    #     x="n_sample_train",
    #     y="runtime_log",
    #     marker="o",
    # )
    # plt.xlabel("Number of subject in model training")
    # plt.ylabel("log10(runtime) (minutes)")
    # plt.title("Runtime of training a group model")
    # plt.savefig("outputs/ccn2024/scaling_runtime_plot.png")
    # plt.close()

    # plt.figure(figsize=(7, 4.5))
    # # plot
    # features = prediction.columns.tolist()
    # for y, label in zip(
    #     features,
    #     [
    #         "connectomes",
    #         "average pooling",
    #         "standard deviation pooling",
    #         "max pooling",
    #         "1D convolution",
    #         "average R-squared",
    #         "R-squared map",
    #     ],
    # ):
    #     if label in [
    #         "connectomes",
    #         "standard deviation pooling",
    #         "R-squared map",
    #     ]:
    #         sns.lineplot(
    #             data=scaling_stats,
    #             x="n_sample_downstream",
    #             y=y,
    #             marker="o",
    #             label=label,
    #         )
    # plt.xlabel("Number of subject in prediction task")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.title("Sex prediction accuracy with SVM")
    # plt.savefig("outputs/ccn2024/_scaling_connectome.png")
    # plt.close()


if __name__ == "__main__":
    main()
