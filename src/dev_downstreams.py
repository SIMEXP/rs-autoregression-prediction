import json
from pathlib import Path
from typing import Dict, List, Union

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from nilearn.connectome import sym_matrix_to_vec
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from src.data.ukbb_datamodule import load_ukbb_dset_path, load_ukbb_sets
from src.data.utils import load_data
from src.models.components.hooks import predict_horizon, save_extracted_feature
from src.models.plotting import plot_horizon
from tqdm import tqdm

CKPT_PATH = "outputs/autoreg/logs/train/multiruns/2024-11-08_08-40-38/0/checkpoints/epoch=059-val_r2_best=0.167.ckpt"
CFG_PATH = "outputs/autoreg/logs/train/multiruns/2024-11-08_08-40-38/0/csv/version_0/hparams.yaml"
CKPT_FEAT = "outputs/autoreg/logs/eval/runs/2024-11-19_15-49-25/features.h5"

measure2data = {
    "connectome_baseline": "connectome",
    "connectome_z": "horizon-all_connectome",
    # "r2map": "r2map",
    # "avgr2": "r2map",
    "conv_max": "layer-3_pooling-max_gcnweights_connectome",
    "conv_avg": "layer-3_pooling-average_gcnweights_connectome",
    "conv_std": "layer-3_pooling-std_gcnweights_connectome",
}


def get_model_data(
    cfg,
    sample_size: float = 0.1,
    measure: str = "connectome",
    label: str = "sex",
) -> Dict[str, np.ndarray]:
    """Get the data from pretrained model for the downstrean task.

    Args:
        data_file (Union[Path, str]): Path to the hdf5 file.
        dset_path (List[str]): List of path to the data inside the
            h5 file.
        phenotype_file (Union[Path, str]): Path to the phenotype file.
        measure (str, optional): Measure to use for the data. Defaults
            to "connectome".
        label (str, optional): Label to use for the data. Defaults to
            "sex".

    Returns:
        Dict[str, np.ndarray]: Dictionary with the data and label.

    Raises:
        NotImplementedError: If the measure is not supported.
    """

    if measure not in [
        "connectome_baseline",
        "connectome_z",
        "r2map",
        "avgr2",
        "conv_max",
        "conv_avg",
        "conv_std",
    ]:
        raise NotImplementedError(
            "measure must be one of 'connectome_baseline', "
            "'connectome_z','r2map', 'avgr2'"
            " or 'conv_max', 'conv_avg', 'conv_std'."
        )
    if label in ["sex", "age"]:
        dset_path = load_ukbb_sets(
            cfg.paths.data_dir, cfg.seed, cfg.data.atlas, stage="test"
        )
    else:
        dset_path = load_ukbb_sets(
            cfg.paths.data_dir, cfg.seed, cfg.data.atlas, stage=label
        )
    participant_id = [
        p.split("/")[-1].split("sub-")[-1].split("_")[0] for p in dset_path
    ]
    n_total = int(len(participant_id) * sample_size)
    df_phenotype = load_phenotype(cfg.data.phenotype_file)
    # get the common subject in participant_id and df_phenotype
    participant_id = list(
        set(participant_id[:n_total]) & set(df_phenotype.index)
    )
    dset_path = dset_path[:n_total]
    # get extra subjects in participant_id
    # remove extra subjects in dset_path
    print(
        f"Subjects with phenotype data: {len(participant_id)}. Total subjects: {n_total}"
    )

    # figure put what extracted feature to load from cfg.features_path
    for p in dset_path:
        subject = p.split("/")[-1].split("sub-")[-1].split("_")[0]
        p = p.replace("timeseries", measure2data[measure])
        if subject in participant_id:
            df_phenotype.loc[subject, "path"] = p
    selected_path = df_phenotype.loc[participant_id, "path"].values.tolist()

    data = []
    with h5py.File(cfg.feature_path, "r") as h5file:
        for p in selected_path:
            d = h5file[p][:]
            if d.shape[-1] == 6:  # last dimension is horizon
                d = d[..., 0]  # use t+1
            if "connectome" in p:
                d = sym_matrix_to_vec(d, discard_diagonal=True)
            elif measure == "avgr2":
                d = d.mean()
            data.append(d)

    labels = np.array(df_phenotype.loc[participant_id, label].values).reshape(
        -1, 1
    )
    data = np.array(data)
    if measure == "avgr2":
        data = data.reshape(-1, 1)
    dataset = {"data": data, "label": labels}
    return dataset


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
        dtype={"participant_id": str, "age": float, "sex": str, "site": str},
    )
    df = df.set_index("participant_id")
    return df


cfg = OmegaConf.load(CFG_PATH)
with open_dict(cfg):
    cfg.paths = {"data_dir": "inputs/data"}
    cfg.feature_path = CKPT_FEAT


def train(cfg, sample, measure):
    data = get_model_data(cfg, sample, measure, "age")
    y = data["label"]
    standard_scalar = StandardScaler()
    x_z = standard_scalar.fit_transform(data["data"])
    # x_z = data['data']
    clf = Lasso(alpha=0.1, random_state=cfg.seed)
    # clf = Ridge(
    #     alpha=0.1,
    #     fit_intercept=True,
    #     copy_X=True,
    #     max_iter=None,
    #     tol=0.0001,
    #     solver="auto",
    #     positive=False,
    #     random_state=cfg.seed,
    # )
    # clf = MLPRegressor(
    #     hidden_layer_sizes=(256, 64),
    #     batch_size=8,
    #     alpha=0.1,
    #     learning_rate_init=1e-3,
    #     max_iter=50
    # )
    rs = ShuffleSplit(n_splits=5, test_size=0.2)
    metrics = []
    for i, (train_index, test_index) in enumerate(rs.split(x_z)):
        clf.fit(x_z[train_index, :], y[train_index, :])
        y_pred = clf.predict(x_z[test_index, :])
        mse = mean_squared_error(y[test_index, :], y_pred)
        mae = mean_absolute_error(y[test_index, :], y_pred)
        r2 = r2_score(y[test_index, :], y_pred)
        print(
            f"metric: {measure} MSE = {mse}, MAE = {mae}, r2 = {r2} N={y.shape[0]}"
        )
        metric = {
            "sample": sample,
            "fold": i + 1,
            "measure": measure,
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }
        metrics.append(metric)
    return metrics


results = Parallel(n_jobs=8, verbose=1, pre_dispatch="1.5*n_jobs")(
    delayed(train)(cfg, s, m)
    for m in measure2data
    for s in [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
)

rr = []
for r in results:
    rr += r

pd.DataFrame.from_dict(rr, orient="columns").to_csv(
    "age_mse_test.tsv", sep="\t"
)

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
