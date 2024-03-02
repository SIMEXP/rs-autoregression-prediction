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
    "connectome": {
        "data_file": None,
        "data_file_pattern": None,
        "plot_label": "Connectome",
    },
    "conv_average": {
        "data_file": None,
        "data_file_pattern": "average",
        "plot_label": "Conv layers \n avg pooling",
    },
    "conv_std": {
        "data_file": None,
        "data_file_pattern": "std",
        "plot_label": "Conv layers \n std pooling",
    },
    "conv_max": {
        "data_file": None,
        "data_file_pattern": "max",
        "plot_label": "Conv layers \n max pooling",
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
    from src.data.load_data import get_model_data, load_h5_data_path

    # parse parameters
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir)
    log.info(f"Output data {output_dir}")
    feature_path = Path(params["feature_path"])
    phenotype_file = Path(params["phenotype_file"])
    convlayers_path = feature_path / "feature_convlayers.h5"
    feature_t1_file = feature_path / f"feature_horizon-{params['horizon']}.h5"
    test_subjects = feature_path / "test_set_connectome.txt"
    model_config = OmegaConf.load(feature_path.parent / ".hydra/config.yaml")
    params = OmegaConf.merge(model_config, params)

    # load test set subject path from the training
    with open(test_subjects, "r") as f:
        subj = f.read().splitlines()

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

    elif params["predict_variable"] == "age":  # need to fix this
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
        # "fold": [],
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
        # sfk = StratifiedKFold(n_splits=5, shuffle=True)
        # folds = sfk.split(dataset["data"], dataset["label"])
        # average_performance = {clf_name: [] for clf_name in clf_names}
        # for i, (tng, tst) in enumerate(folds, start=1):
        #     log.info(f"Fold {i}")
        #     for clf_name, clf in zip(clf_names, [svm, lr, rr, mlp]):
        #         clf.fit(dataset["data"][tng], dataset["label"][tng])
        #         score = clf.score(
        #             dataset["data"][tst], dataset["label"][tst]
        #         )
        #         log.info(
        #             f"{measure} - {clf_name} fold {i}: {score:.3f}"
        #         )
        #         baselines_df["feature"].append(measure)
        #         baselines_df["score"].append(score)
        #         baselines_df["classifier"].append(clf_name)
        #         baselines_df["fold"].append(i)
        #         average_performance[clf_name].append(score)
        # for clf_name in clf_names:
        #     acc = np.mean(average_performance[clf_name])
        #     log.info(
        #         f"{measure} - {clf_name} average score: {acc:.3f}"
        #     )

        tng, tst = next(
            StratifiedKFold(n_splits=5, shuffle=True).split(
                dataset["data"], dataset["label"]
            )
        )  # only one fold
        for clf_name, clf in zip(clf_names, [svm, lr, rr, mlp]):
            clf.fit(dataset["data"][tng], dataset["label"][tng])
            score = clf.score(dataset["data"][tst], dataset["label"][tst])
            log.info(f"{measure} - {clf_name} score: {score:.3f}")
            baselines_df["feature"].append(measure)
            baselines_df["score"].append(score)
            baselines_df["classifier"].append(clf_name)

    # save the results
    # json for safe keeping
    with open(
        output_dir / f"simple_classifiers_{params['predict_variable']}.json",
        "w",
    ) as f:
        json.dump(baselines_df, f, indent=4)

    baselines_df = pd.DataFrame(baselines_df)
    baselines_df.to_csv(
        output_dir / f"simple_classifiers_{params['predict_variable']}.tsv",
        sep="\t",
    )


if __name__ == "__main__":
    main()
