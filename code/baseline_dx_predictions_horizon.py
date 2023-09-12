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
from giga_companion.load_data import load_data, load_h5_data_path
from nilearn.connectome import ConnectivityMeasure
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

data_file = "inputs/connectomes/processed_connectomes.h5"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", "-m", type=Path, help="model output directory"
    )
    parser.add_argument(
        "--horizon_dir", "-p", type=Path, help="Path to horizon predictiondir."
    )
    args = parser.parse_args()

    (args.horizon_dir / "figures").mkdir(exist_ok=True)

    model_name = args.model_dir.name

    feature_t1_file = args.horizon_dir / f"{model_name}_horizon-1.h5"

    # ABIDE 1
    abide1 = load_h5_data_path(
        data_file,
        "abide1.*/*/sub-.*desc-197.*timeseries",
        shuffle=True,
    )
    # ABIDE 2
    abide2 = load_h5_data_path(
        data_file,
        "abide2.*/*/sub-.*desc-197.*timeseries",
        shuffle=True,
    )

    # get connectomes
    cm = ConnectivityMeasure(kind="correlation", vectorize=True)
    tng_data = cm.fit_transform(
        [load_data(data_file, d, dtype="data")[0] for d in abide1]
    )
    tng_label = [load_data(data_file, d, dtype="diagnosis")[0] for d in abide1]
    tng_sites = [d.split("/")[1] for d in abide1]

    test_data = cm.fit_transform(
        [load_data(data_file, d, dtype="data")[0] for d in abide2]
    )
    test_label = [
        load_data(data_file, d, dtype="diagnosis")[0] for d in abide2
    ]

    # three baseline models
    svc = LinearSVC(C=100, penalty="l2", max_iter=100000, random_state=42)
    lr = LogisticRegression(penalty="l2", max_iter=100000, random_state=42)
    rr = RidgeClassifier(random_state=42, max_iter=100000)

    # connectome
    print("Train on ABIDE 1, Test on ABIDE 2")
    print("Connectome")
    acc = []
    for clf in [svc, lr, rr]:
        clf.fit(tng_data, tng_label)
        test_pred = clf.predict(test_data)
        acc.append(accuracy_score(test_label, test_pred))
        print(f"accuracy: {acc[-1]:.3f}")

    baseline = pd.DataFrame(
        {
            "Accuracy": acc,
            "Classifier": ["SVC", "LR", "Ridge"],
            "Feature": ["Connectome", "Connectome", "Connectome"],
        }
    )
    # r2 map
    print("r2 map")
    r2_path = load_h5_data_path(feature_t1_file, "r2map")
    tng_path = [p for p in r2_path if "abide1" in p]
    test_path = [p for p in r2_path if "abide2" in p]

    tng_label = [
        load_data(feature_t1_file, p, dtype="diagnosis") for p in tng_path
    ]
    test_label = [
        load_data(feature_t1_file, p, dtype="diagnosis") for p in test_path
    ]

    tng_data = [load_data(feature_t1_file, p, dtype="data") for p in tng_path]
    tng_data = np.concatenate(tng_data).squeeze()
    z_tng_data = StandardScaler().fit_transform(tng_data)
    test_data = [
        load_data(feature_t1_file, p, dtype="data") for p in test_path
    ]
    test_data = np.concatenate(test_data).squeeze()
    z_test_data = StandardScaler().fit_transform(test_data)

    acc = []
    for clf in [svc, lr, rr]:
        clf.fit(z_tng_data, tng_label)
        test_pred = clf.predict(z_test_data)
        acc.append(accuracy_score(test_label, test_pred))
        print(f"accuracy: {acc[-1]:.3f}")
    basline_r2map = pd.DataFrame(
        {
            "Accuracy": acc,
            "Classifier": ["SVC", "LR", "Ridge"],
            "Feature": ["R2 map", "R2 map", "R2 map"],
        }
    )

    # mean r2
    print("average r2")
    acc = []
    avg_tng_data = tng_data.mean(axis=1).reshape(-1, 1)
    avg_tng_data = StandardScaler().fit_transform(avg_tng_data)
    avg_test_data = test_data.mean(axis=1).reshape(-1, 1)
    avg_test_data = StandardScaler().fit_transform(avg_test_data)
    for clf in [svc, lr, rr]:
        clf.fit(avg_tng_data, tng_label)
        test_pred = clf.predict(avg_test_data)
        acc.append(accuracy_score(test_label, test_pred))
        print(f"accuracy: {acc[-1]:.3f}")
    basline_avgr2 = pd.DataFrame(
        {
            "Accuracy": acc,
            "Classifier": ["SVC", "LR", "Ridge"],
            "Feature": ["Average R2", "Average R2", "Average R2"],
        }
    )

    # plotting
    df = pd.concat([baseline, basline_avgr2, basline_r2map], axis=0)
    df.to_csv(args.horizon_dir / "simple_classifiers.csv")

    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(6, 5))
    sns.despine(bottom=True, left=True)
    sns.pointplot(
        data=df,
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
        loc="lower right",
        ncol=1,
        frameon=True,
        columnspacing=1,
        handletextpad=0,
    )
    plt.ylim(0.3, 0.9)
    plt.hlines(0.5, -0.3, 3, linestyles="dashed", colors="black")
    plt.tight_layout()
    plt.savefig(args.horizon_dir / "figures/simple_classifiers.png")
