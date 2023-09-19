"""
Execute at the root of the repo, not in the code directory.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from giga_companion.load_data import load_data, load_h5_data_path
from nilearn.connectome import ConnectivityMeasure
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# get connectomes
def get_model_data(
    data_file, tng_dset_path, test_dset_path, measure="connectome"
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

    dataset = {}
    for set_name, dset_path in zip(
        ["tng", "test"], [tng_dset_path, test_dset_path]
    ):
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

        label = load_data(data_file, dset_path, dtype="diagnosis")

        dataset[set_name] = {"data": data, "label": label}
    return dataset


baseline_details = {
    "connectome": {
        "data_file": "inputs/connectomes/processed_connectomes.h5",
        "data_file_pattern": "abide.*/*/sub-.*desc-197.*timeseries",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", "-m", type=Path, help="model output directory"
    )
    parser.add_argument(
        "--feature_dir", "-f", type=Path, help="Path to horizon predictiondir."
    )
    args = parser.parse_args()

    (args.model_dir / "figures").mkdir(exist_ok=True)

    model_name = args.model_dir.name

    feature_t1_file = args.feature_dir / f"{model_name}_horizon-1.h5"
    convlayer_file = args.feature_dir / f"{model_name}_convlayers.h5"
    print(feature_t1_file)
    print(convlayer_file)
    for key in baseline_details:
        if "r2" in key:
            baseline_details[key]["data_file"] = feature_t1_file
        elif "conv" in key:
            baseline_details[key]["data_file"] = convlayer_file
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

    print("Train on ABIDE 1, Test on ABIDE 2")
    for measure in baseline_details:
        print(measure)
        dset_path = load_h5_data_path(
            baseline_details[measure]["data_file"],
            baseline_details[measure]["data_file_pattern"],
            shuffle=True,
        )
        tng_path = [p for p in dset_path if "abide1" in p]
        test_path = [p for p in dset_path if "abide2" in p]
        dataset = get_model_data(
            baseline_details[measure]["data_file"],
            tng_dset_path=tng_path,
            test_dset_path=test_path,
            measure=measure,
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

    # plotting
    baselines_df = pd.concat(
        baselines_df,
        axis=0,
    )
    baselines_df.to_csv(
        args.feature_dir / "figures/simple_classifiers.tsv", sep="\t"
    )

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
    plt.title(
        "Baseline test accuracy\ntraining set: ABIDE 1, test set: ABIDE 2"
    )
    plt.tight_layout()
    plt.savefig(args.feature_dir / "figures/simple_classifiers.png")
