import logging
import os

# import rootutils
import warnings

import hydra
import lightning as L
from lightning import LightningDataModule
from omegaconf import DictConfig

# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train.yaml"
)
def main(cfg: DictConfig) -> None:
    in_job_array: str = "SLURM_ARRAY_TASK_ID" in os.environ
    if in_job_array:
        array_task_id: int = int(os.environ["SLURM_ARRAY_TASK_ID"])
        array_task_count: int = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        log.info(
            f"This job is at index {array_task_id} in a job array of size {array_task_count}"
        )

    if cfg.verbose.get("ignore_warnings"):
        log.info(
            "Disabling python warnings! <cfg.verbose.ignore_warnings=True>"
        )
        warnings.filterwarnings("ignore")
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    log.info("Finishing...")


if __name__ == "__main__":
    main()
