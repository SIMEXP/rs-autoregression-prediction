"""
Extract features from the model.
If model was trained on gpu, this script should
be run on a machine with a gpu.
"""
from typing import Any, Dict, List, Optional, Tuple

import h5py
import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from tqdm import tqdm

torch._dynamo.config.suppress_errors = True  # work around for triton issue

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.data.ukbb_datamodule import load_ukbb_sets
from src.data.utils import load_data
from src.models.autoreg_module import GraphAutoRegModule
from src.models.components.hooks import predict_horizon
from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


# need to implement
# https://docs.h5py.org/en/latest/mpi.html
@task_wrapper
def extract(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    I can't figure out how to plug this in to pytorch prediction
    hook so here's a stand alone script.
    """

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    assert cfg.ckpt_path

    log.info(f"Instantiating pytorch model <{cfg.model._target_}>")
    cp = torch.load(cfg.ckpt_path)
    if "net" not in cp["hyper_parameters"]:
        model: LightningModule = GraphAutoRegModule.load_from_checkpoint(
            cfg.ckpt_path, net=hydra.utils.instantiate(cfg.model.net)
        )
    else:
        model: LightningModule = GraphAutoRegModule.load_from_checkpoint(
            cfg.ckpt_path
        )
    model.freeze()  # put in evaluation mode

    log.info("Loading test set subjects")
    conn_dset_paths: List = load_ukbb_sets(
        cfg.paths.data_dir, cfg.seed, cfg.data.atlas, stage="test"
    )

    log.info("Extract features.")
    # save to data dir
    output_extracted_feat: str = f"{cfg.paths.output_dir}/features.h5"
    edge = torch.tensor(hydra.utils.instantiate(cfg.model.edge_index)).to(
        model.device
    )
    with h5py.File(output_extracted_feat, "a") as f:
        for dset_path in tqdm(conn_dset_paths):
            data_tmp: List | None = load_data(
                cfg.data.connectome_file, dset_path
            )
            if data_tmp is None:
                continue

            data: np.ndarray = data_tmp[0]
            del data_tmp
            # make labels per subject
            predict_horizon(
                cfg.horizon,
                model,
                data,
                edge_index=edge,
                timeseries_window_stride_lag=cfg.data.timeseries_window_stride_lag,
                timeseries_decimate=cfg.data.timeseries_decimate,
                f=f,
                dset_path=dset_path,
            )
            del data
            f.flush()
    log.info("Extraction completed.")
    return None, None  # competibility with task wrapper


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    # extract features
    extract(cfg)


if __name__ == "__main__":
    main()
