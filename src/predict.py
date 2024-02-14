"""
Execute at the root of the repo, not in the code directory.
"""
import argparse
import json
import logging
from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

baseline_details = {
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
}

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../config", config_name="predict")
def main(params: DictConfig) -> None:
    from src.data.load_data import get_model_data, load_data, load_h5_data_path

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
    log.info(f"Predicting {params['predict_variable']}.")

    if params["predict_variable"] == "sex":
        # four baseline models for sex
        svm = LinearSVC(C=100, penalty="l2", max_iter=1000000, random_state=42)
        lr = LogisticRegression(
            penalty="l2", max_iter=100000, random_state=42, n_jobs=-1
        )
        rr = RidgeClassifier(random_state=42, max_iter=100000)
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            max_iter=100000,
            random_state=42,
        )
        clf_names = ["SVM", "LogisticR", "Ridge", "MLP"]

    elif params["predict_variable"] == "age":
        # four baseline models for age
        svm = LinearSVR(C=100, max_iter=1000000, random_state=42)
        lr = LinearRegression(n_jobs=-1)
        rr = Ridge(random_state=42, max_iter=100000)
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            max_iter=100000,
            random_state=42,
        )
        clf_names = ["SVM", "LinearR", "Ridge", "MLP"]
    else:
        raise ValueError("predict_variable must be either sex or age")

    baselines_df = {
        "feature": [],
        "score": [],
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
            label=params["predict_variable"],
            log=log,
        )
        log.info("Start training...")
        sfk = StratifiedKFold(n_splits=5, shuffle=True)
        folds = sfk.split(dataset["data"], dataset["label"])
        average_performance = {clf_name: [] for clf_name in clf_names}
        for i, (tng, tst) in enumerate(folds, start=1):
            log.info(f"Fold {i}")
            for clf_name, clf in zip(clf_names, [svm, lr, rr, mlp]):
                clf.fit(dataset["data"][tng], dataset["label"][tng])
                score = clf.score(dataset["data"][tst], dataset["label"][tst])
                log.info(f"{measure} - {clf_name} fold {i} score: {score:.3f}")
                baselines_df["feature"].append(measure)
                baselines_df["score"].append(score)
                baselines_df["classifier"].append(clf_name)
                baselines_df["fold"].append(i)
                average_performance[clf_name].append(score)
            if i == 1:
                break
        # for clf_name in clf_names:
        #     acc = np.mean(average_performance[clf_name])
        #     log.info(
        #         f"{measure} - {clf_name} average score: {acc:.3f}"
        #     )

    # save the results
    # json for safe keeping
    with open(
        output_dir / f"simple_classifiers_{params['predict_variable']}.json",
        "w",
    ) as f:
        json.dump(baselines_df, f, indent=4)

    baselines_df = pd.DataFrame(baselines_df)
    baselines_df.to_csv(
        output_dir,
        f"simple_classifiers_{params['predict_variable']}.tsv",
        sep="\t",
    )

    # plt.figure()
    # sns.set_theme(style="whitegrid")
    # f, ax = plt.subplots(figsize=(9, 5))
    # sns.despine(bottom=True, left=True)
    # sns.pointplot(
    #     data=baselines_df,
    #     x="Feature",
    #     y="Accuracy",
    #     hue="Classifier",
    #     join=False,
    #     dodge=0.4 - 0.4 / 3,
    #     markers="d",
    #     scale=0.75,
    #     errorbar=None,
    # )
    # sns.move_legend(
    #     ax,
    #     loc="upper right",
    #     ncol=1,
    #     frameon=True,
    #     columnspacing=1,
    #     handletextpad=0,
    # )
    # plt.ylim(0.4, 1.0)
    # plt.hlines(0.5, -0.5, 5.5, linestyles="dashed", colors="black")
    # plt.title("Baseline test accuracy")
    # plt.tight_layout()
    # plt.savefig(output_dir / "simple_classifiers_sex.png")


if __name__ == "__main__":
    main()
