"""
Execute at the root of the repo, not in the code directory.
"""
import argparse
import json
import logging
from pathlib import Path

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from nilearn.connectome import ConnectivityMeasure
from omegaconf import DictConfig, OmegaConf
from seaborn import boxplot
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

baseline_details = {
    "connectome": {
        "data_file": None,
        "data_file_pattern": None,
        "plot_label": "Connectome",
    },
    "avgr2": {
        "data_file": None,
        "data_file_pattern": "r2map",
        "plot_label": "t+1\n average R2",
    },
    "r2map": {
        "data_file": None,
        "data_file_pattern": "r2map",
        "plot_label": "t+1\nR2 map",
    },
    "conv_avg": {
        "data_file": None,
        "data_file_pattern": "convlayers",
        "plot_label": "Conv layers \n avg pooling",
    },
    "conv_std": {
        "data_file": None,
        "data_file_pattern": "convlayers",
        "plot_label": "Conv layers \n std pooling",
    },
    "conv_max": {
        "data_file": None,
        "data_file_pattern": "convlayers",
        "plot_label": "Conv layers \n max pooling",
    },
}

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="predict")
def main(params: DictConfig) -> None:
    from src.data.load_data import load_data, load_h5_data_path

    # get connectomes
    def get_model_data(
        data_file, dset_path, phenotype_file, measure="connectome", label="sex"
    ):
        if measure not in [
            "connectome",
            "r2map",
            "avgr2",
            "conv_max",
            "conv_avg",
            "conv_std",
        ]:
            raise NotImplementedError(
                "measure must be one of 'connectome', 'r2map', 'avgr2'"
            )
        if measure == "connectome":
            cm = ConnectivityMeasure(kind="correlation", vectorize=True)
        elif measure == "conv_max":
            m = nn.AdaptiveMaxPool3d((1, 1, 1))
        elif measure == "conv_avg":
            m = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif measure == "conv_std":
            m = lambda x: torch.std_mean(x, (1, 2, 3))[0]

        participant_id = [
            p.split("/")[-1].split("sub-")[-1].split("_")[0] for p in dset_path
        ]
        n_total = len(participant_id)
        df_phenotype = pd.read_csv(
            phenotype_file,
            sep="\t",
            dtype={"participant_id": str, "age": float, "sex": int},
        )
        df_phenotype = df_phenotype.set_index("participant_id")
        # get the common subject in participant_id and df_phenotype
        participant_id = list(set(participant_id) & set(df_phenotype.index))
        # get extra subjects in participant_id
        # remove extra subjects in dset_path
        dset_path = [
            p
            for p in dset_path
            if p.split("/")[-1].split("sub-")[-1].split("_")[0]
            in participant_id
        ]
        log.info(
            f"Subjects with phenotype data: {len(participant_id)}. Total subjects: {n_total}"
        )

        # 1:4 split dset_path
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tng_idx, test_idx = next(
            skf.split(participant_id, df_phenotype.loc[participant_id, label])
        )
        dataset = {}
        for set_name, subjects in zip(["tng", "test"], [tng_idx, test_idx]):
            dset_path = [
                p
                for p in dset_path
                if p.split("/")[-1].split("sub-")[-1].split("_")[0] in subjects
            ]
            data = load_data(data_file, dset_path, dtype="data")
            if "r2" in measure:
                data = np.concatenate(data).squeeze()
                if measure == "avgr2":
                    data = data.mean(axis=1).reshape(-1, 1)
                data = StandardScaler().fit_transform(data)

            if measure == "connectome":
                data = cm.fit_transform(data)

            if "conv" in measure:
                convlayers = []
                for d in data:
                    d = torch.tensor(d, dtype=torch.float32)
                    d = m(d).flatten().numpy()
                    convlayers.append(d)
                data = convlayers
            label = df_phenotype.loc[subjects, label]
            dataset[set_name] = {"data": data, "label": label}
        return dataset

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    convlayers_path = params["convlayers_path"]
    feature_t1_file = (
        convlayers_path.parent / f"feature_horizon-{params['horizon']}.h5"
    )
    model_path = Path(params["model_path"])
    phenotype_file = Path(params["phenotype_file"])

    log.info(feature_t1_file)
    log.info(convlayers_path)

    # load test set subject path from the training
    with open(model_path.parent / "train_test_split.json", "r") as f:
        subj = json.load(f)

    for key in baseline_details:
        if "r2" in key:
            baseline_details[key]["data_file"] = feature_t1_file
        elif "conv" in key:
            baseline_details[key]["data_file"] = convlayers_path
        elif "connectome" in key:
            baseline_details[key]["data_file"] = params["data"]["data_file"]
            baseline_details[key]["data_file_pattern"] = subj["test"]
        else:
            pass

    # four baseline models
    svc = LinearSVC(C=100, penalty="l2", max_iter=1000000, random_state=42)
    lr = LogisticRegression(penalty="l2", max_iter=100000, random_state=42)
    rr = RidgeClassifier(random_state=42, max_iter=100000)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        max_iter=100000,
        random_state=42,
    )
    clf_names = ["SVC", "LR", "Ridge", "MLP (sklearn)"]

    baselines_df = []

    for measure in baseline_details:
        print(measure)
        if measure == "connectome":
            dset_path = baseline_details[measure]["data_file_pattern"]
        else:
            dset_path = load_h5_data_path(
                baseline_details[measure]["data_file"],
                baseline_details[measure]["data_file_pattern"],
                shuffle=True,
            )

        dataset = get_model_data(
            baseline_details[measure]["data_file"],
            dset_path=dset_path,
            phenotype_file=phenotype_file,
            measure=measure,
            label="sex",
        )
        acc = []
        for clf_name, clf in zip(clf_names, [svc, lr, rr, mlp]):
            clf.fit(dataset["tng"]["data"], dataset["tng"]["label"])
            test_pred = clf.predict(dataset["test"]["data"])
            acc_score = accuracy_score(dataset["test"]["label"], test_pred)
            acc.append(acc_score)
            print(f"{clf_name} accuracy: {acc_score:.3f}")

        baseline = pd.DataFrame(
            {
                "Accuracy": acc,
                "Classifier": clf_names,
                "Feature": [baseline_details[measure]["plot_label"]] * 4,
            }
        )
        baselines_df.append(baseline)

    # stats_path = (
    #     params["feature_path"] /
    #      f"feature_horizon-{params['horizon']}.tsv"
    # )
    # print("Plot r2 mean results quickly.")
    # # quick plotting
    # df_for_stats = pd.read_csv(stats_path, sep="\t")
    # df_for_stats = df_for_stats[
    #     (df_for_stats.r2mean > -1e16)
    # ]  # remove outliers
    # plt.figure()
    # g = boxplot(x="site", y="r2mean",
    #             hue="diagnosis", data=df_for_stats)
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    # plt.savefig(output_dir / "diagnosis_by_sites.png")

    # plt.figure()
    # g = boxplot(x="site", y="r2mean", hue="sex", data=df_for_stats)
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    # plt.savefig(output_dir / "sex_by_sites.png")

    # plotting
    baselines_df = pd.concat(
        baselines_df,
        axis=0,
    )
    baselines_df.to_csv(output_dir, "simple_classifiers.tsv", sep="\t")

    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(9, 5))
    sns.despine(bottom=True, left=True)
    sns.pointplot(
        data=baselines_df,
        x="Feature",
        y="Accuracy",
        hue="Classifier",
        join=False,
        dodge=0.4 - 0.4 / 3,
        markers="d",
        scale=0.75,
        errorbar=None,
    )
    sns.move_legend(
        ax,
        loc="upper right",
        ncol=1,
        frameon=True,
        columnspacing=1,
        handletextpad=0,
    )
    plt.ylim(0.4, 0.7)
    plt.hlines(0.5, -0.5, 5.5, linestyles="dashed", colors="black")
    plt.title("Baseline test accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "simple_classifiers.png")


if __name__ == "__main__":
    main()
