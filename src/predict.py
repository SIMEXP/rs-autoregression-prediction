"""
Execute at the root of the repo, not in the code directory.

```
python src/predict.py --multirun \
  feature_path=/path/to/<name of your analysis>/extract \
```

Currently the script hard coded to predict sex or age.
"""
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

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
    "conv_conv1d": {
        "data_file": None,
        "data_file_pattern": "1dconv",
        "plot_label": "Conv layers \n 1D convolution",
    },
}

log = logging.getLogger(__name__)


def train(dataset, tng, tst, clf, clf_name):
    clf.fit(dataset["data"][tng], dataset["label"][tng])
    score = clf.score(dataset["data"][tst], dataset["label"][tst])
    return {
        "score": score,
        "classifier": clf_name,
    }


LABEL_DIR = Path(__file__).parents[1] / "outputs" / "sample_for_pretraining"


@hydra.main(version_base="1.3", config_path="../config", config_name="predict")
def main(params: DictConfig) -> None:
    from src.data.load_data import (
        get_model_data,
        load_h5_data_path,
        load_ukbb_dset_path,
    )

    # parse parameters
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir)
    log.info(f"Output data {output_dir}")
    feature_path = Path(params["feature_path"])
    extract_config = OmegaConf.load(feature_path / ".hydra/config.yaml")
    model_config = OmegaConf.load(
        Path(extract_config["model_path"]).parent / ".hydra/config.yaml"
    )

    phenotype_file = Path(model_config["data"]["phenotype_file"])
    convlayers_path = feature_path / "feature_convlayers.h5"
    feature_t1_file = (
        feature_path / f"feature_horizon-{extract_config['horizon']}.h5"
    )
    params = OmegaConf.merge(model_config, params)
    log.info(params)
    percentage_sample = params["percentage_sample"]

    if params["predict_variable"] in ["age", "sex"]:
        sample_file = list(
            (LABEL_DIR / f"seed-{model_config['random_state']}").glob(
                "sample*split.json"
            )
        )[0]

        # load test set subject path from the training
        with open(sample_file, "r") as f:
            hold_outs = json.load(f)[f"{model_config['data']['n_embed']}"][
                "test"
            ]

        if percentage_sample != 100:
            proportion = percentage_sample / 100
            sample_select = ShuffleSplit(
                n_splits=1,
                train_size=proportion,
                random_state=params["random_state"],
            )
            sample_index, _ = next(sample_select.split(hold_outs))
            subj = [hold_outs[i] for i in sample_index]
        else:
            subj = hold_outs.copy()
    else:
        sample_file = (
            LABEL_DIR
            / f"seed-{model_config['random_state']}"
            / "downstream_sample.json"
        )
        with open(sample_file, "r") as f:
            hold_outs = json.load(f)["test_downstreams"][
                params["predict_variable"]
            ]  # these are subject ids
        diagnosis_data = (
            pd.read_csv(phenotype_file, sep="\t")
            .set_index("participant_id")
            .loc[hold_outs, :]
        )

        percentage_sample = params["percentage_sample"]
        if percentage_sample != 100:
            proportion = percentage_sample / 100
            sample_select = StratifiedShuffleSplit(
                n_splits=1,
                train_size=proportion,
                random_state=params["random_state"],
            )
            sample_index, _ = next(
                sample_select.split(
                    diagnosis_data.index,
                    diagnosis_data[params["predict_variable"]],
                )
            )
            subj = [diagnosis_data.index[i] for i in sample_index]

        else:
            subj = hold_outs.copy()
        subj = [f"sub-{s}" for s in subj]
        subj = load_ukbb_dset_path(
            subj, params["data"]["atlas_desc"], params["data"]["segment"]
        )

    log.info(
        f"Downstream prediction on {len(subj)}, "
        f"{percentage_sample}% of the full sample."
    )

    for key in baseline_details:
        if "r2" in key:
            baseline_details[key]["data_file"] = feature_t1_file
        elif "conv" in key:
            baseline_details[key]["data_file"] = convlayers_path
        elif "connectome" in key:
            baseline_details[key]["data_file"] = params["data"]["data_file"]
            baseline_details[key]["data_file_pattern"] = subj
        else:
            pass
    log.info(f"Predicting {params['predict_variable']}.")

    if params["predict_variable_type"] == "binary":
        clf_options = {
            "SVM": LinearSVC(
                C=100,
                penalty="l2",
                class_weight="balanced",
                max_iter=10000,
                random_state=params["random_state"],
            ),
            "LogisticR": LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=1000,
                random_state=params["random_state"],
                n_jobs=-1,
            ),
            "Ridge": RidgeClassifier(
                class_weight="balanced",
                random_state=params["random_state"],
                max_iter=1000,
            ),
        }
    elif params["predict_variable_type"] == "numerical":  # need to fix this
        clf_options = {
            "SVM": LinearSVR(
                C=100, max_iter=10000, random_state=params["random_state"]
            ),
            "LinearR": LinearRegression(n_jobs=-1),
            "Ridge": Ridge(random_state=params["random_state"], max_iter=1000),
        }
    else:
        raise ValueError(
            "predict_variable_type must be either binary or numerical"
        )

    df_results = []
    for measure in baseline_details:
        log.info(f"Start training {measure}")
        log.info(f"Load data {baseline_details[measure]['data_file']}")
        if measure == "connectome":
            dset_path = baseline_details[measure]["data_file_pattern"]
        else:
            dset_path = []
            for connectome_path in subj:
                dp = connectome_path.replace(
                    "timeseries",
                    baseline_details[measure]["data_file_pattern"],
                )
                dset_path.append(dp)

        dataset = get_model_data(
            baseline_details[measure]["data_file"],
            dset_path=dset_path,
            phenotype_file=phenotype_file,
            measure=measure,
            label=params["predict_variable"],
            log=log,
        )
        log.info(f"found {len(dataset['data'])} subjects with {measure} data.")
        log.info("Start training...")

        sfk = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=params["random_state"]
        )
        out = Parallel(n_jobs=4, verbose=100, pre_dispatch="1.5*n_jobs")(
            delayed(train)(dataset, tng, tst, clf_options[key], key)
            for key in clf_options
            for tng, tst in sfk.split(dataset["data"], dataset["label"])
        )
        df_out = pd.DataFrame(out)
        df_out["feature"] = measure
        df_results.append(df_out)

        df_out = df_out.groupby("classifier").mean()
        for clf_name, acc in zip(df_out.index, df_out["score"]):
            log.info(f"{measure} - {clf_name} average score: {acc:.3f}")

    df_results = pd.concat(df_results).reset_index(drop=True)
    df_results.to_csv(
        output_dir / f"simple_classifiers_{params['predict_variable']}.tsv",
        sep="\t",
    )


if __name__ == "__main__":
    main()
