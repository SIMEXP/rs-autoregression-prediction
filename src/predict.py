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
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

baseline_details = {
    # "avgr2": {
    #     "data_file": None,
    #     "data_file_pattern": "r2map",
    #     "plot_label": "t+1\n average R2",
    # },
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
    "connectome": {
        "data_file": None,
        "data_file_pattern": None,
        "plot_label": "Connectome",
    },
}

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="predict")
def main(params: DictConfig) -> None:
    from data.load_data import get_model_data, load_data, load_h5_data_path

    # parse parameters
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    convlayers_path = Path(params["convlayers_path"])
    feature_t1_file = (
        convlayers_path.parent / f"feature_horizon-{params['horizon']}.h5"
    )
    model_path = Path(params["model_path"])
    phenotype_file = Path(params["phenotype_file"])

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

    # four baseline models for sex
    svc = LinearSVC(C=100, penalty="l2", max_iter=1000000, random_state=42)
    lr = LogisticRegression(penalty="l2", max_iter=100000, random_state=42)
    rr = RidgeClassifier(random_state=42, max_iter=100000)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        max_iter=100000,
        random_state=42,
    )
    clf_names = ["SVC", "LR", "Ridge", "MLP (sklearn)"]

    baselines_df = {
        "feature": [],
        "accuracy": [],
        "classifier": [],
        "fold": [],
    }
    for measure in baseline_details:
        log.info(f"Start training {measure}")
        log.info(f"Load data {baseline_details[measure]['data_file']}")
        if measure == "connectome":
            dset_path = baseline_details[measure]["data_file_pattern"]
        else:
            dset_path = load_h5_data_path(
                baseline_details[measure]["data_file"],
                baseline_details[measure]["data_file_pattern"],
                shuffle=True,
                random_state=params["random_state"],
            )
        log.info(f"found {len(dset_path)} subjects with {measure} data.")

        dataset = get_model_data(
            baseline_details[measure]["data_file"],
            dset_path=dset_path,
            phenotype_file=phenotype_file,
            measure=measure,
            label="sex",
            log=log,
        )

        sfk = StratifiedKFold(n_splits=5, shuffle=True)
        folds = sfk.split(dataset["data"], dataset["label"])
        average_performance = {clf_name: [] for clf_name in clf_names}
        for i, (tng, tst) in enumerate(folds, start=1):
            for clf_name, clf in zip(clf_names, [svc, lr, rr, mlp]):
                clf.fit(dataset["data"][tng], dataset["label"][tng])
                test_pred = clf.predict(dataset["data"][tst])
                acc_score = accuracy_score(dataset["label"][tst], test_pred)
                log.info(
                    f"{measure} - {clf_name} fold {i} accuracy: {acc_score:.3f}"
                )
                baselines_df["feature"].append(measure)
                baselines_df["accuracy"].append(acc_score)
                baselines_df["classifier"].append(clf_name)
                baselines_df["fold"].append(i)
                average_performance[clf_name].append(acc_score)
        for clf_name in clf_names:
            acc = np.mean(average_performance[clf_name])
            log.info(f"{measure} - {clf_name} average accuracy: {acc:.3f}")

    plt.figure()

    # plotting
    baselines_df = pd.DataFrame(baselines_df)
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
    plt.ylim(0.4, 1.0)
    plt.hlines(0.5, -0.5, 5.5, linestyles="dashed", colors="black")
    plt.title("Baseline test accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "simple_classifiers_sex.png")


if __name__ == "__main__":
    main()
