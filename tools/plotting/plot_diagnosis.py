import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

DIAGNOSIS_PATH = "outputs/neuroips-workshop_2024/downstreams_last-layer/data/data.proportion_sample_1.0"  # noqa: E501

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
    "MS": "Multiple\nsclerosis",
    "PARK": "Parkinson",
    "BIPOLAR": "Bipolar",
    "ADD": "Alzheimer -\nDementia",
    "SCZ": "Schizophrenia",
}


def main():
    diagnosis_path = Path(DIAGNOSIS_PATH)
    diagnosis_files = diagnosis_path.glob("**/*.tsv")
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    fig, axs = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    n_subjects_diagnosis = {}
    for ax, classifier in zip(axs, ["SVM", "LogisticR", "Ridge"]):
        # classifier = "LogisticR"
        data_clf = []
        diagnosis_files = diagnosis_path.glob("**/*.tsv")
        for p in diagnosis_files:
            filename = p.name
            diagnosis = filename.split("_")[-1].split(".")[0]
            df = pd.read_csv(p, sep="\t")
            df = df.loc[df.classifier == classifier, :]
            df = df.groupby("feature")["score"].agg("mean").reset_index()
            if diagnosis != "sex":
                with open(p.parent / "predict.log", "r") as f:
                    log = f.read()
                    n_subjects = (
                        int(
                            re.search(
                                r"Downstream prediction on ([\d]*),", log
                            ).group(1)
                        )
                        / 2
                    )
                n_subjects_diagnosis[diagnosis] = int(n_subjects)
            else:
                with open(p.parent / "predict.log", "r") as f:
                    log = f.read()
                    n_holdout = int(
                        re.search(
                            r"Downstream prediction on ([\d]*),", log
                        ).group(1)
                    )
                n_subjects_diagnosis[
                    diagnosis
                ] = 3341  # number of male subjects
            df["diagnosis"] = diagnosis_fullname[diagnosis]
            data_clf.append(df)
        data_clf = pd.concat(data_clf)
        data_clf = data_clf.reset_index(drop=True)

        # for each diagnosis, get index of results better than connectome
        idx_better = []
        for _, diagnosis in enumerate(diagnosis_fullname.values()):
            baseline = data_clf.loc[
                (data_clf.feature == "connectome")
                & (data_clf.diagnosis == diagnosis),
                "score",
            ].values[0]
            baseline_idx = data_clf.loc[
                (data_clf.feature == "connectome")
                & (data_clf.diagnosis == diagnosis),
                "score",
            ].index[0]
            better = (
                data_clf.loc[
                    (data_clf.feature != "connectome")
                    & (data_clf.diagnosis == diagnosis),
                    "score",
                ]
                >= baseline
            )
            better = (
                data_clf.loc[
                    (data_clf.feature != "connectome")
                    & (data_clf.diagnosis == diagnosis),
                    "score",
                ]
                .index[better]
                .tolist()
            )
            better.append(baseline_idx)
            idx_better += better
        idx_better.sort()
        # get index that is the opposite of idx_better
        idx_better = np.array(idx_better)
        idx = np.zeros(data_clf.shape[0], dtype=bool)
        idx[idx_better] = True
        sns.stripplot(
            x="diagnosis",
            y="score",
            hue="feature",
            data=data_clf.iloc[~idx, :],
            ax=ax,
            legend=False,
            order=diagnosis_fullname.values(),
            hue_order=feature_fullname.keys(),
            marker="$\circ$",
            size=10,
            jitter=0.2,
        )
        sns.stripplot(
            x="diagnosis",
            y="score",
            hue="feature",
            data=data_clf.iloc[idx, :],
            ax=ax,
            legend=classifier == "Ridge",
            order=diagnosis_fullname.values(),
            hue_order=feature_fullname.keys(),
            size=8,
            jitter=0.25,
        )

        ax.hlines(y=0.5, xmin=0.5, xmax=8.5, color="k", linestyle="--")
        ax.hlines(
            y=n_subjects_diagnosis["sex"] / n_holdout,
            xmin=-0.5,
            xmax=0.5,
            color="k",
            linestyle="--",
        )
        ax.set_title(f"{classifier}")
        ax.set_ylim(0.4, 1)
        ax.set_ylabel("Accuracy score")
        tick_lables = []
        for tl in diagnosis_fullname:
            if tl == "sex":
                tick_lables.append(
                    tl.upper()
                    + " ($N_{male}=$"
                    + f"${n_subjects_diagnosis[tl]}$)"
                )
            else:
                tick_lables.append(tl + f" ($N={n_subjects_diagnosis[tl]}$)")

        ax.set_xticklabels(tick_lables, rotation=90)
        chance = Line2D([0], [0], color="black", label="Chance", ls="--")
        # get legend handles and labels
        han, lab = axs[-1].get_legend_handles_labels()
        han.append(chance)
        legend_labels = [feature_fullname[i] for i in lab]
        legend_labels.append("Chance")
        # append cahnce line to the legend
        axs[-1].legend(handles=han, labels=legend_labels)
        sns.move_legend(axs[-1], "upper left", bbox_to_anchor=(1, 1))
        fig.suptitle(
            "Downstream prediction (Training set proportion = "
            f"{DIAGNOSIS_PATH.split('_')[-1].replace('-', ' ')})"
        )
        # fig.suptitle(f"Downstream prediction on Full hold out sample)")
        plt.tight_layout()
        plt.savefig(
            Path(DIAGNOSIS_PATH).parents[1]
            / "reports"
            / f"{Path(DIAGNOSIS_PATH).name}_overview_LR.png"
        )


if __name__ == "__main__":
    main()
