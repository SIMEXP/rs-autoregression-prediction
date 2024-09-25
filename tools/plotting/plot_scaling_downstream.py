"""
look through the `outputs/` directory, find instance of completed
training, and get the number of subjects used, mean R2 of test set,
plot the number of subjects (y-axis) against R2 (x axis)
"""
import itertools
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


feature_fullname = {
    "connectome": "Connectome\n(baseline)",
    "avgr2": "t+1\naverage R2",
    "r2map": "t+1\nR2 map",
    "conv_avg": "Conv layers\navg pooling",
    "conv_std": "Conv layers\nstd pooling",
    "conv_max": "Conv layers\nmax pooling",
    "conv_conv1d": "Conv layers\n1D convolution",
}

diagnosis_fullname = {
    "sex": "Sex",
    "DEP": "Depressive\ndisorder",
    "ALCO": "Alcohol Abuse",
    "EPIL": "Epilepsy",
    "MS": "Multiple sclerosis",
    "PARK": "Parkinson",
    "BIPOLAR": "Bipolar",
    "ADD": "Alzheimer - Dementia",
    "SCZ": "Schizophrenia",
}
PREDICTION_DATA = Path(
    "outputs/neuroips-workshop_2024/downstreams_last-layer/data"
)


def main():
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    pal = sns.color_palette()
    for d in diagnosis_fullname.keys():
        path_success_job = PREDICTION_DATA.glob(
            f"data.proportion_sample_*/**/simple_classifiers_{d}.tsv"
        )
        scaling_stats = pd.DataFrame()
        for p in path_success_job:
            log_file = p.parent / "predict.log"
            with open(log_file, "r") as f:
                log_text = f.read()
            n_holdout = int(
                re.search(
                    r"Downstream prediction on ([\d]*),", log_text
                ).group(1)
            )
            # parse the path and get number of subjects
            percent_training_sample = p.parents[0].name.split("_")[-1]
            percent_training_sample = float(percent_training_sample) * 100
            percent_holdout_sample = re.search(
                r"([\d]*)% of the full sample", log_text
            ).group(1)
            # get random seed
            random_seed = re.search(
                r"'random\_state': ([\d]+)", log_text
            ).group(1)
            # load connectome accuracy
            prediction = pd.read_csv(p, sep="\t", index_col=0)
            prediction["percent_holdout_sample"] = int(percent_holdout_sample)
            prediction["percent_training_sample"] = percent_training_sample
            prediction["random_seed"] = random_seed

            scaling_stats = pd.concat([scaling_stats, prediction], axis=0)

        # sort by n_sample
        scaling_stats = scaling_stats.sort_values(by="percent_training_sample")
        # # for each n_sample, sort by random seed
        # scaling_stats = scaling_stats.groupby("percent_training_sample").apply(
        #     lambda x: x.sort_values(by="random_seed")
        # )
        scaling_stats = scaling_stats.reset_index(drop=True)

        scaling_stats.to_csv(
            PREDICTION_DATA.parent / f"reports/downstream_scaling_{d}.tsv",
            sep="\t",
        )
        # replace feature name
        scaling_stats["percent_training_sample"] = np.log10(
            scaling_stats["percent_training_sample"]
        )
        scaling_stats["feature"] = scaling_stats["feature"].replace(
            feature_fullname
        )
        for clf in scaling_stats["classifier"].unique():
            mask = scaling_stats["classifier"] == clf
            no_connectome = (
                scaling_stats["feature"] != "Connectome\n(baseline)"
            )
            is_connectome = ~no_connectome
            plt.figure(figsize=(7, 4.5))
            benchmark = scaling_stats[mask & is_connectome]["score"].mean()
            plt.axhline(
                y=benchmark, color=pal[0], linestyle="-.", label="Connectome"
            )
            # plot
            features = prediction["feature"].unique().tolist()
            sns.lineplot(
                data=scaling_stats[mask & no_connectome],
                x="percent_training_sample",
                y="score",
                hue="feature",
                marker="o",
                errorbar="ci",
                palette=pal[1 : len(features)],
            )
            if d != "sex":
                plt.axhline(y=0.5, color="k", linestyle="--", label="Chance")
            else:
                plt.axhline(
                    y=3341 / n_holdout,
                    color="k",
                    linestyle="--",
                    label="Chance",
                )
            plt.xlabel("Percent of subject in the pretrained model")
            plt.xticks(scaling_stats["percent_training_sample"].unique())
            plt.ylabel("Accuracy")
            plt.legend(bbox_to_anchor=(1, 1))
            plt.title(
                f"{diagnosis_fullname[d]} prediction accuracy with {clf}"
            )
            plt.tight_layout()
            plt.savefig(
                PREDICTION_DATA.parent
                / f"reports/downstream_scaling_{d}_{clf}.png"
            )
            plt.close()


if __name__ == "__main__":
    main()
