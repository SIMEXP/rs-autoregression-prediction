import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Union

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
from nilearn.image import math_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import (
    find_xyz_cut_coords,
    plot_img_on_surf,
    plot_stat_map,
)
from omegaconf import OmegaConf, open_dict
from scipy.stats import zscore
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    Lasso,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from src.data.ukbb_datamodule import load_ukbb_dset_path, load_ukbb_sets
from src.data.utils import load_data
from src.models.components.hooks import predict_horizon, save_extracted_feature
from src.models.plotting import plot_horizon
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# CKPT_PATH = "outputs/model_registery/current_best.ckpt"
# CFG_PATH = "outputs/model_registery/current_best.yaml"
# CKPT_FEAT = (
#     "outputs/features/current_best/features.h5"
# )
# CKPT_PATH = "outputs/model_registery/baseline.ckpt"
# CFG_PATH = "outputs/model_registery/baseline.yaml"
# CKPT_FEAT = (
#     "outputs/features/baseline/features.h5"
# )

measure2data = {
    "connectome_baseline": "connectome",
    "connectome_z": "horizon-all_connectome",
    "r2map": "r2map",
    "avgr2": "r2map",
}
# measures = [
#     "connectome_baseline",
#     "connectome_z",
#     "r2map",
#     "layer-NonsharedFC1_pooling-average_weights",
#     "layer-NonsharedFC1_pooling-max_weights",
#     "layer-NonsharedFC1_pooling-std_weights",
#     "layer-ChebConv3_pooling-average_weights",
#     "layer-ChebConv3_pooling-max_weights",
#     "layer-ChebConv3_pooling-std_weights",
#     "layer-NonsharedFC1_connectome",
#     "layer-ChebConv3_connectome",
# ]
logger = logging.getLogger("downstream")


def measure_name_validator(measure):
    if measure in [
        "connectome_baseline",
        "connectome_z",
        "r2map",
        "avgr2",
    ]:
        return measure2data[measure]
    pool_weight_pattern = r"layer-[A-z\d]*_pooling-[a-z]*_weights"
    pool_results = re.match(pool_weight_pattern, measure)
    conn_weight_pattern = r"layer-[A-z\d]*_connectome"
    conn_results = re.match(conn_weight_pattern, measure)
    if pool_results is None and conn_results is None:
        raise NotImplementedError(
            "measure must be one of 'connectome_baseline', "
            "'connectome_z','r2map', 'avgr2'"
            " or in the format of 'layer-{layername}{layernumber}_pooling-{pooling}_weights',"
            " or 'layer-{layername}{layernumber}_connectome'."
        )
    elif conn_results is None:
        return measure
    else:
        return measure.replace("connectome", "weights")


def prepare_model_data(
    cfg,
    data_dir,
    measure: str = "connectome",
    phenotype: str = "sex",
    sample: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Get the data from pretrained model for the downstrean task.

    Args:
        data_file (Union[Path, str]): Path to the hdf5 file.
        dset_path (List[str]): List of path to the data inside the
            h5 file.
        phenotype_file (Union[Path, str]): Path to the phenotype file.
        measure (str, optional): Measure to use for the data. Defaults
            to "connectome".

    Returns:
        Dict[str, np.ndarray]: Dictionary with the data and label.

    Raises:
        NotImplementedError: If the measure is not supported.
    """

    cleaned_measure = measure_name_validator(measure)

    if phenotype in ["sex", "age"]:
        dset_path = load_ukbb_sets(
            data_dir, cfg.seed, cfg.data.atlas, stage="test"
        )
    else:
        dset_path = load_ukbb_sets(
            data_dir, cfg.seed, cfg.data.atlas, stage=phenotype
        )
    n_subject = int(sample * len(dset_path))
    dset_path = dset_path[:n_subject]
    participant_id = [
        p.split("/")[-1].split("sub-")[-1].split("_")[0] for p in dset_path
    ]
    df_phenotype = load_phenotype(cfg.data.phenotype_file)

    # get the common subject in participant_id and df_phenotype
    participant_id = list(set(participant_id) & set(df_phenotype.index))
    # figure put what extracted feature to load from cfg.features_path
    for p in dset_path:
        subject = p.split("/")[-1].split("sub-")[-1].split("_")[0]
        p = p.replace("timeseries", cleaned_measure)
        if subject in participant_id:
            df_phenotype.loc[subject, "path"] = p
    # remove data with no path value
    return df_phenotype.dropna(axis=0, subset="path")


def load_phenotype(path: Union[Path, str]) -> pd.DataFrame:
    """Load the phenotype data from the file.

    Args:
        path (Union[Path, str]): Path to the phenotype file.

    Returns:
        pd.DataFrame: Phenotype data.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "participant_id": str,
            "age": float,
            "sex": int,
            "site": str,
            "DEP": int,
            "ALCO": int,
            "EPIL": int,
            "MS": int,
            "PARK": int,
            "BIPOLAR": int,
            "ADD": int,
            "SCZ": int,
        },
    )
    df = df.set_index("participant_id")
    return df


def load_brain_features(feature_path, labels, measure):
    selected_path = labels["path"].values.tolist()
    data = []
    with h5py.File(feature_path, "r") as h5file:
        for p in selected_path:
            try:
                d = h5file[p][:]
            except KeyError:
                logger.info(f"missing {p}")

            if d.shape[-1] == 6:  # last dimension should always be horizon
                d = d[..., 0]  # use t+1
            if "connectome" in p:
                d = sym_matrix_to_vec(d, discard_diagonal=True)
            elif measure == "avgr2":
                d = d.mean()
            elif "pooling" in measure:
                d = d.flatten()
            elif "layer" in p:
                conn = ConnectivityMeasure(
                    kind="correlation", vectorize=True, discard_diagonal=True
                )
                d = conn.fit_transform([d[..., f] for f in range(d.shape[-1])])
                d = np.mean(d, axis=0)  # average over filters
            data.append(d)
    return pd.DataFrame(np.array(data), index=selected_path)


def load_training_data(all_labels, all_brain, phenotype):
    y = all_labels.loc[:, phenotype].values
    x = zscore(all_brain.loc[:, :].values, axis=1)
    return x, y


def train(x, y, i, train_index, test_index, phenotype, measure, sample, clf):
    clf.fit(x[train_index, :], y[train_index])
    y_pred = clf.predict(x[test_index, :])

    if phenotype == "age":
        mse = mean_squared_error(y[test_index], y_pred)
        mae = mean_absolute_error(y[test_index], y_pred)
        r2 = r2_score(y[test_index], y_pred)
        print(
            f"{phenotype} metric: {measure} MSE = {mse}, MAE = {mae}, "
            f"r2 = {r2} N={y.shape[0]}"
        )
        metric = {
            "sample": sample,
            "fold": i + 1,
            "measure": measure,
            "phenotype": phenotype,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }
    else:
        metric = {
            "sample": sample,
            "fold": i + 1,
            "measure": measure,
            "phenotype": phenotype,
            "accuracy": accuracy_score(y[test_index], y_pred),
            "f1": f1_score(y[test_index], y_pred),
            "auc": roc_auc_score(y[test_index], y_pred),
        }
        print(
            f"{phenotype} metric: {measure} ACC = {metric['accuracy']}, F1 = {metric['f1']}, AUC = {metric['auc']} N={y.shape[0]}"
        )
    return metric


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="downstream.yaml"
)
def main(cfg):
    measure = cfg.measure
    phenotype = cfg.phenotype
    sample = cfg.proportion_sample
    model_cfg = OmegaConf.load(cfg.model_cfg)
    output_dir_name = cfg.model_cfg.split("/")[-1].split(".")[0]
    Path(f"outputs/downstream/{output_dir_name}").mkdir(exist_ok=True)

    results = []
    all_labels = prepare_model_data(
        cfg=model_cfg,
        data_dir=cfg.paths.data_dir,
        measure=measure,
        phenotype=phenotype,
        sample=sample,
    )  # always use sex or age to load the full set
    all_brain = load_brain_features(cfg.feature_path, all_labels, measure)
    assert all_labels.shape[0] == all_brain.shape[0]
    x, y = load_training_data(all_labels, all_brain, phenotype=phenotype)
    if phenotype == "age":
        clf = Lasso(alpha=0.1, random_state=model_cfg.seed)
    else:
        clf = LogisticRegression(
            penalty="l2", class_weight="balanced", random_state=model_cfg.seed
        )
    if phenotype != "age":
        rs = StratifiedShuffleSplit(
            n_splits=5, test_size=0.2, random_state=model_cfg.seed
        )
        if sample < 0.2 or phenotype != "sex":
            rs = StratifiedShuffleSplit(
                n_splits=10, test_size=0.1, random_state=model_cfg.seed
            )
        results = Parallel(n_jobs=8, verbose=1, pre_dispatch="1.5*n_jobs")(
            delayed(train)(
                x,
                y,
                i,
                train_index,
                test_index,
                phenotype,
                measure,
                sample,
                clf,
            )
            for i, (train_index, test_index) in enumerate(rs.split(x, y))
        )
    else:
        rs = ShuffleSplit(
            n_splits=5, test_size=0.2, random_state=model_cfg.seed
        )
        if sample < 0.2:
            rs = ShuffleSplit(
                n_splits=10, test_size=0.1, random_state=model_cfg.seed
            )
        results = Parallel(n_jobs=8, verbose=1, pre_dispatch="1.5*n_jobs")(
            delayed(train)(
                x,
                y,
                i,
                train_index,
                test_index,
                phenotype,
                measure,
                sample,
                clf,
            )
            for i, (train_index, test_index) in enumerate(rs.split(x))
        )
    pd.DataFrame.from_dict(results, orient="columns").to_csv(
        f"outputs/downstream/{output_dir_name}/downstream_{phenotype}_{measure}_{sample*100}.tsv",
        sep="\t",
    )


if __name__ == "__main__":
    main()

# """
# Execute at the root of the repo, not in the code directory.

# ```
# python src/predict.py --multirun \
#   feature_path=/path/to/<name of your analysis>/extract \
# ```

# Currently the script hard coded to predict sex or age.
# """
# import json
# import logging
# from pathlib import Path

# import hydra
# import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed
# from omegaconf import DictConfig, OmegaConf
# from sklearn.linear_model import (
#     LinearRegression,
#     LogisticRegression,
#     Ridge,
#     RidgeClassifier,
# )
# from sklearn.model_selection import (
#     ShuffleSplit,
#     StratifiedKFold,
#     StratifiedShuffleSplit,
# )
# from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.svm import LinearSVC, LinearSVR

# baseline_details = {
#     "connectome": {
#         "data_file": None,
#         "data_file_pattern": None,
#         "plot_label": "Connectome",
#     },
#     "avgr2": {
#         "data_file": None,
#         "data_file_pattern": "r2map",
#         "plot_label": "t+1\n average R2",
#     },
#     "r2map": {
#         "data_file": None,
#         "data_file_pattern": "r2map",
#         "plot_label": "t+1\nR2 map",
#     },
#     "conv_avg": {
#         "data_file": None,
#         "data_file_pattern": "average",
#         "plot_label": "Conv layers \n avg pooling",
#     },
#     "conv_std": {
#         "data_file": None,
#         "data_file_pattern": "std",
#         "plot_label": "Conv layers \n std pooling",
#     },
#     "conv_max": {
#         "data_file": None,
#         "data_file_pattern": "max",
#         "plot_label": "Conv layers \n max pooling",
#     },
#     "conv_conv1d": {
#         "data_file": None,
#         "data_file_pattern": "1dconv",
#         "plot_label": "Conv layers \n 1D convolution",
#     },
# }

# log = logging.getLogger(__name__)


# def train(dataset, tng, tst, clf, clf_name):
#     clf.fit(dataset["data"][tng], dataset["label"][tng])
#     score = clf.score(dataset["data"][tst], dataset["label"][tst])
#     return {
#         "score": score,
#         "classifier": clf_name,
#     }


# LABEL_DIR = Path(__file__).parents[1] / "outputs" / "sample_for_pretraining"


# @hydra.main(version_base="1.3", config_path="../config", config_name="predict")
# def main(params: DictConfig) -> None:
#     from src.data.load_data import (
#         get_model_data,
#         load_h5_data_path,
#         load_ukbb_dset_path,
#     )

#     # parse parameters
#     output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
#     output_dir = Path(output_dir)
#     log.info(f"Output data {output_dir}")
#     feature_path = Path(params["feature_path"])
#     extract_config = OmegaConf.load(feature_path / ".hydra/config.yaml")
#     model_config = OmegaConf.load(
#         Path(extract_config["model_path"]).parent / ".hydra/config.yaml"
#     )

#     phenotype_file = Path(model_config["data"]["phenotype_file"])
#     convlayers_path = feature_path / "feature_convlayers.h5"
#     feature_t1_file = (
#         feature_path / f"feature_horizon-{extract_config['horizon']}.h5"
#     )
#     params = OmegaConf.merge(model_config, params)
#     log.info(params)
#     percentage_sample = params["percentage_sample"]

#     if params["predict_variable"] in ["age", "sex"]:
#         sample_file = list(
#             (LABEL_DIR / f"seed-{model_config['random_state']}").glob(
#                 "sample*split.json"
#             )
#         )[0]

#         # load test set subject path from the training
#         with open(sample_file, "r") as f:
#             hold_outs = json.load(f)[f"{model_config['data']['n_embed']}"][
#                 "test"
#             ]

#         if percentage_sample != 100:
#             proportion = percentage_sample / 100
#             sample_select = ShuffleSplit(
#                 n_splits=1,
#                 train_size=proportion,
#                 random_state=params["random_state"],
#             )
#             sample_index, _ = next(sample_select.split(hold_outs))
#             subj = [hold_outs[i] for i in sample_index]
#         else:
#             subj = hold_outs.copy()
#     else:
#         sample_file = (
#             LABEL_DIR
#             / f"seed-{model_config['random_state']}"
#             / "downstream_sample.json"
#         )
#         with open(sample_file, "r") as f:
#             hold_outs = json.load(f)["test_downstreams"][
#                 params["predict_variable"]
#             ]  # these are subject ids
#         diagnosis_data = (
#             pd.read_csv(phenotype_file, sep="\t")
#             .set_index("participant_id")
#             .loc[hold_outs, :]
#         )

#         percentage_sample = params["percentage_sample"]
#         if percentage_sample != 100:
#             proportion = percentage_sample / 100
#             sample_select = StratifiedShuffleSplit(
#                 n_splits=1,
#                 train_size=proportion,
#                 random_state=params["random_state"],
#             )
#             sample_index, _ = next(
#                 sample_select.split(
#                     diagnosis_data.index,
#                     diagnosis_data[params["predict_variable"]],
#                 )
#             )
#             subj = [diagnosis_data.index[i] for i in sample_index]

#         else:
#             subj = hold_outs.copy()
#         subj = [f"sub-{s}" for s in subj]
#         subj = load_ukbb_dset_path(
#             subj, params["data"]["atlas_desc"], params["data"]["segment"]
#         )

#     log.info(
#         f"Downstream prediction on {len(subj)}, "
#         f"{percentage_sample}% of the full sample."
#     )

#     for key in baseline_details:
#         if "r2" in key:
#             baseline_details[key]["data_file"] = feature_t1_file
#         elif "conv" in key:
#             baseline_details[key]["data_file"] = convlayers_path
#         elif "connectome" in key:
#             baseline_details[key]["data_file"] = params["data"]["data_file"]
#             baseline_details[key]["data_file_pattern"] = subj
#         else:
#             pass
#     log.info(f"Predicting {params['predict_variable']}.")

#     if params["predict_variable_type"] == "binary":
#         clf_options = {
#             "SVM": LinearSVC(
#                 C=100,
#                 penalty="l2",
#                 class_weight="balanced",
#                 max_iter=10000,
#                 random_state=params["random_state"],
#             ),
#             "LogisticR": LogisticRegression(
#                 penalty="l2",
#                 class_weight="balanced",
#                 max_iter=1000,
#                 random_state=params["random_state"],
#                 n_jobs=-1,
#             ),
#             "Ridge": RidgeClassifier(
#                 class_weight="balanced",
#                 random_state=params["random_state"],
#                 max_iter=1000,
#             ),
#         }
#     elif params["predict_variable_type"] == "numerical":  # need to fix this
#         clf_options = {
#             "SVM": LinearSVR(
#                 C=100, max_iter=10000, random_state=params["random_state"]
#             ),
#             # "LinearR": LinearRegression(n_jobs=-1), # just skip this, not use
#             "Ridge": Ridge(random_state=params["random_state"], max_iter=1000),
#         }
#     else:
#         raise ValueError(
#             "predict_variable_type must be either binary or numerical"
#         )

#     df_results = []
#     for measure in baseline_details:
#         log.info(f"Start training {measure}")
#         log.info(f"Load data {baseline_details[measure]['data_file']}")
#         if measure == "connectome":
#             dset_path = baseline_details[measure]["data_file_pattern"]
#         else:
#             dset_path = []
#             for connectome_path in subj:
#                 dp = connectome_path.replace(
#                     "timeseries",
#                     baseline_details[measure]["data_file_pattern"],
#                 )
#                 dset_path.append(dp)

#         dataset = get_model_data(
#             baseline_details[measure]["data_file"],
#             dset_path=dset_path,
#             phenotype_file=phenotype_file,
#             measure=measure,
#             label=params["predict_variable"],
#             log=log,
#         )
#         log.info(f"found {len(dataset['data'])} subjects with {measure} data.")
#         log.info("Start training...")

#         sfk = StratifiedKFold(
#             n_splits=5, shuffle=True, random_state=params["random_state"]
#         )
#         out = Parallel(n_jobs=4, verbose=100, pre_dispatch="1.5*n_jobs")(
#             delayed(train)(dataset, tng, tst, clf_options[key], key)
#             for key in clf_options
#             for tng, tst in sfk.split(dataset["data"], dataset["label"])
#         )
#         df_out = pd.DataFrame(out)
#         df_out["feature"] = measure
#         df_results.append(df_out)

#         df_out = df_out.groupby("classifier").mean()
#         for clf_name, acc in zip(df_out.index, df_out["score"]):
#             log.info(f"{measure} - {clf_name} average score: {acc:.3f}")

#     df_results = pd.concat(df_results).reset_index(drop=True)
#     df_results.to_csv(
#         output_dir / f"simple_classifiers_{params['predict_variable']}.tsv",
#         sep="\t",
#     )


# if __name__ == "__main__":
#     main()
