"""
Extract features from the model.
If model was trained on gpu, this script should
be run on a machine with a gpu.
"""
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import h5py
import hydra
import lightning as L
import numpy as np
import rootutils
import torch
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
from src.models.components.hooks import predict_horizon, save_extracted_feature
from src.models.utils import load_model_from_ckpt
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

    log.info(f"Instantiating pytorch model <{cfg.model.net._target_}>")
    net: torch.nn.Module = load_model_from_ckpt(
        cfg.ckpt_path, hydra.utils.instantiate(cfg.model.net)
    )
    net.eval()  # put in evaluation mode
    log.info("Loading test set subjects")
    conn_dset_paths: List = load_ukbb_sets(
        cfg.paths.data_dir, cfg.seed, cfg.data.atlas, stage="test"
    )

    log.info("Extract features.")
    # save to data dir
    # output_extracted_feat: str = f"{cfg.paths.data_dir}/features.h5"
    output_extracted_feat: str = f"{cfg.paths.output_dir}/features.h5"

    f: h5py.File = h5py.File(output_extracted_feat, "a")
    for dset_path in tqdm(conn_dset_paths):
        data: np.ndarray = load_data(cfg.data.connectome_file, dset_path)[0]
        # make labels per subject
        y, z, convlayers, ts_r2 = predict_horizon(
            cfg.horizon,
            net,
            data,
            edge_index=torch.tensor(
                hydra.utils.instantiate(cfg.model.edge_index)
            ),
            timeseries_window_stride_lag=cfg.data.timeseries_window_stride_lag,
            timeseries_decimate=cfg.data.timeseries_decimate,
        )

        # save to h5 file, create more features
        save_extracted_feature(
            data[::4], y, z, convlayers, ts_r2, f, dset_path
        )
        del y
        del z
        del convlayers
        del ts_r2
    log.info("Extraction completed.")
    f.close()
    log.info("Copy to project.")
    # # copy from data dir to output dir
    output_extracted_feat: str = f"{cfg.paths.output_dir}/features.h5"
    subprocess.run(
        [
            "rsync",
            "-tv",
            "--info=progress2",
            output_extracted_feat,
            f"{cfg.paths.output_dir}/features.h5",
        ]
    )
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
