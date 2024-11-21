import json
from typing import Dict, List, Union

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from nilearn.image import math_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import (
    find_xyz_cut_coords,
    plot_img_on_surf,
    plot_stat_map,
)
from omegaconf import OmegaConf, open_dict
from scipy.stats import zscore
from src.data.ukbb_datamodule import load_ukbb_dset_path
from src.data.utils import load_data
from src.models.components.hooks import predict_horizon, save_extracted_feature
from src.models.plotting import plot_horizon

CKPT_PATH = "outputs/autoreg/logs/train/multiruns/2024-11-08_08-40-38/0/checkpoints/epoch=059-val_r2_best=0.167.ckpt"
CFG_PATH = "outputs/autoreg/logs/train/multiruns/2024-11-08_08-40-38/0/csv/version_0/hparams.yaml"
MIST197_PCC = 30  # PCC in mist 197: 30
PCC_COORDS = (0, -45, 20)
output_extracted_feat = "outputs/test.h5"
MIST197 = "inputs/atlas/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_atlas-MIST_res-dataset_desc-197_dseg.nii.gz"
horizon = 6


cfg = OmegaConf.load(CFG_PATH)
with open_dict(cfg):
    cfg.paths = {"data_dir": "inputs/data"}

# load the model
# from src.models.autoreg_module import GraphAutoRegModule
# model = GraphAutoRegModule.load_from_checkpoint(CKPT_PATH)
ckpt = torch.load(CKPT_PATH)
model_weights = ckpt["state_dict"]
for key in list(model_weights):  # hack to use the model directly
    model_weights[key.replace("net.", "")] = model_weights.pop(key)
net = hydra.utils.instantiate(cfg.model.net)
net.load_state_dict(model_weights)
net.eval()  # put in evaluation mode

# get testing set paths
path_sample_split = (
    f"{cfg.paths.data_dir}/downstream_sample_seed-{cfg.seed}.json"
)
with open(path_sample_split, "r") as f:
    subj_list = json.load(f)["test"]
conn_dset_paths = load_ukbb_dset_path(
    participant_id=subj_list,
    atlas_desc=f"atlas-{cfg.data.atlas[0]}_desc-{cfg.data.atlas[1]}",
    segment=0,  # use full time series, decimate later in make_sequence_single_subject
)

f = h5py.File(output_extracted_feat, "w")

dset_path = conn_dset_paths[0]  # load just one subject
data = load_data(cfg.data.connectome_file, dset_path)[
    0
]  # remember this is the original TR

# make labels per subject
y, z, convlayers, ts_r2 = predict_horizon(
    horizon,
    net,
    data,
    edge_index=torch.tensor(hydra.utils.instantiate(cfg.model.edge_index)),
    timeseries_window_stride_lag=cfg.data.timeseries_window_stride_lag,
    timeseries_decimate=cfg.data.timeseries_decimate,
)
# save to h5 file, create more features
save_extracted_feature(data[::4], y, z, convlayers, ts_r2, f, dset_path)

f.close()

# conn = ConnectivityMeasure(kind='correlation')
# for h in range(horizon):
#     # functional connectivity from simulated signal
#     y_conn, z_conn = conn.fit_transform([y[:, :, h], z[:, :, h]])
#     conn_r2 = r2_score(y_conn, z_conn, multioutput="raw_values")
#     last_layer_weights = np.moveaxis(convlayers[h][-1], 0, -1)  # (# ROI, # outputs nodes, # time windows)
#     last_layer_weights = torch.tensor(last_layer_weights).unsqueeze(0)  # 0 dimension is batch
#     _, n_parcel, n_output_nodes, n_time = last_layer_weights.shape
#     pooling_funcs = generate_pooling_funcs(n_parcel, n_output_nodes, kernel_time=1)

#     pooled = {}
#     for f in pooling_funcs:
#         d = pooling_funcs[f](last_layer_weights)
#     # if h == 0 :
#     #     # peak = np.argmax(ts_r2)
#     #     # peak_parcel = math_img(f'img == {peak+1}', img=MIST197)
#     #     # peak_coord = find_xyz_cut_coords(peak_parcel)
#     #     # seed_name = 'R2peak'
#     #     peak = MIST197_PCC-1
#     #     peak_parcel = math_img(f'img == {MIST197_PCC}', img=MIST197)
#     #     peak_coord = PCC_COORDS
#     #     seed_name = 'PCC'
#     #     print(ts_r2[peak])
#     #     mist197 = NiftiLabelsMasker(MIST197).fit()
#     #     # plot r^2 map on atlas, pcc seed based y_conn map and z_conn map
#     #     nii_ts_r2 = mist197.inverse_transform(ts_r2)
#     #     nii_peak_y = mist197.inverse_transform(y_conn[:, peak])
#     #     peak_on_real_data = y[:, :, h].copy()
#     #     peak_on_real_data[:, peak] = z[:, peak, h]
#     #     z_conn = conn.fit_transform([peak_on_real_data])[0]
#     #     nii_peak_z = mist197.inverse_transform(z_conn[:, peak])

#         # get all kinds of pooling for the last layer
#         last_layer_weights = np.moveaxis(convlayers[h][-1], 0, -1)  # (# ROI, # outputs nodes, # time windows)
#         last_layer_weights = torch.tensor(last_layer_weights).unsqueeze(0)  # 0 dimension is batch
#         _, n_parcel, n_output_nodes, n_time = last_layer_weights.shape
#         pooling_funcs = generate_pooling_funcs(n_parcel, n_output_nodes, kernel_time=1)

#         pooled = {}
#         for f in pooling_funcs:
#             d = pooling_funcs[f](last_layer_weights)
#             print(d.shape)
#             if d.ndim == 2:
#                 weights_conn = conn.fit_transform([d.T])[0]
#                 nii_peak_w = mist197.inverse_transform(weights_conn[:, peak])
#             else:
#                 nii_peak_w = mist197.inverse_transform(d)
#             pooled[f] = (d, nii_peak_w)
#             plt.figure()
#             plot_img_on_surf(
#                 nii_peak_w,
#                 views=["lateral", "medial"],
#                 hemispheres=["left", "right"],
#                 colorbar=True,
#                 title=f'{seed_name} connectivity of\ngcn weights (pooling: {f}).',
#                 bg_on_data=True,
#             )
#             plt.savefig(f'outputs/gcn_weights_last_{f}_{seed_name}_lag-{h+1}.png')
#             plt.close()

#     #     # get all kinds of pooling for all layer
#     #     last_layer_weights = np.moveaxis(np.concatenate(convlayers[0], axis=-1), 0, -1)  # (# ROI, # outputs nodes, # time windows)
#     #     last_layer_weights = torch.tensor(last_layer_weights).unsqueeze(0)  # 0 dimension is batch
#     #     _, n_parcel, n_output_nodes, n_time = last_layer_weights.shape
#     #     pooling_funcs = generate_pooling_funcs(n_parcel, n_output_nodes, kernel_time=n_output_nodes)
#     #     pooled = {}
#     #     for f in pooling_funcs:
#     #         d = pooling_funcs[f](last_layer_weights)
#     #         weights_conn = conn.fit_transform([d.T])[0]
#     #         nii_peak_w = mist197.inverse_transform(weights_conn[:, peak])
#     #         pooled[f] = (d, nii_peak_w)
#     #         plt.figure()
#     #         plot_img_on_surf(
#     #             nii_peak_w,
#     #             views=["lateral", "medial"],
#     #             hemispheres=["left", "right"],
#     #             colorbar=True,
#     #             title=f'{seed_name} connectivity of\ngcn weights (pooling: {f})',
#     #             bg_on_data=True,
#     #         )
#     #         plt.savefig(f'outputs/gcn_weights_all_{f}_{seed_name}_lag-{h+1}.png')
#     #         plt.close()

#     #     # plotting
#     #     plt.figure()
#     #     plot_img_on_surf(
#     #         nii_ts_r2,
#     #         views=["lateral", "medial"],
#     #         hemispheres=["left", "right"],
#     #         colorbar=True,
#     #         title=f'R2 map of t+1 prediction of {conn_dset_paths[0].split("/")[2]}.',
#     #         bg_on_data=True,
#     #     )
#     #     plt.savefig(f'outputs/r2_lag-{h+1}.png')
#     #     plt.close()


#     #     plt.figure()
#     #     plot_img_on_surf(
#     #         nii_peak_y,
#     #         views=["lateral", "medial"],
#     #         hemispheres=["left", "right"],
#     #         colorbar=True,
#     #         title=f'{seed_name} connectivity\noriginal data of {conn_dset_paths[0].split("/")[2]}',
#     #         bg_on_data=True,
#     #     )
#     #     plt.savefig(f'outputs/seed_{seed_name}.png')

#     #     plt.figure()
#     #     plot_img_on_surf(
#     #         nii_peak_z,
#     #         views=["lateral", "medial"],
#     #         hemispheres=["left", "right"],
#     #         colorbar=True,
#     #         title=f'{seed_name} connectivity\nsimulated t+1 data of {conn_dset_paths[0].split("/")[2]}',
#     #         bg_on_data=True,
#     #     )
#     #     plt.savefig(f'outputs/seed_{seed_name}_lag-{h+1}.png')
#     #     plt.close()


# full_ts = data[::4]
# plot_horizon(
#     full_ts,
#     predict_ts=z,
#     window_size=w,
#     horizon=horizon,
#     seeds={'pcc': MIST197_PCC-1, 'peak': np.argmax(horizon_ts_r2[0])}
# )
