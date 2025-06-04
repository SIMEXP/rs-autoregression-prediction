from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
import torch
from lightning import LightningModule
from omegaconf import DictConfig

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

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)

import torch
from calflops import calculate_flops
from src.models.autoreg_module import GraphAutoRegModule

batch_size = 128
window_size = 16
n_parcel = 197


@task_wrapper
def extract(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    edge = torch.tensor(hydra.utils.instantiate(cfg.model.edge_index))
    if "ckpt_path" in cfg:
        cp = torch.load(cfg.ckpt_path)
        if "net" not in cp["hyper_parameters"]:
            model: LightningModule = GraphAutoRegModule.load_from_checkpoint(
                cfg.ckpt_path, net=hydra.utils.instantiate(cfg.model.net)
            )
        else:
            model: LightningModule = GraphAutoRegModule.load_from_checkpoint(
                cfg.ckpt_path
            )
    else:
        model: LightningModule = GraphAutoRegModule(
            net=hydra.utils.instantiate(cfg.model)
        )

    input_shape = (
        batch_size,
        n_parcel,
        window_size,
    )
    x = torch.ones(()).new_empty(
        (*input_shape,), dtype=next(model.parameters()).dtype
    )
    flops, macs, params = calculate_flops(
        model=model,
        output_as_string=False,
        print_results=False,
        output_unit=None,
        kwargs={"x": x, "edge_index": edge},
    )
    name = cfg.ckpt_path.split("/")[-1].split(".")[0]
    log.info(f"FLOPs:{flops}   MACs:{macs}   Params:{params} \n")
    # save this in a tsv
    with open(f"outputs/performance_info/{name}.txt", "w") as f:
        f.write("name\tflops\tmacs\tparams\n")
        f.write(f"{name}\t{flops}\t{macs}\t{params}\n")
    return None, None


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train.yaml"
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
