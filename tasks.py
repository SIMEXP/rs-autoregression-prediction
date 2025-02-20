import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import invoke
import numpy as np
import pandas as pd
import yaml
from comet_ml import API
from dotenv import load_dotenv
from tqdm import tqdm


@invoke.task
def download_models(c):
    load_dotenv()
    api = API(os.getenv("COMET_API_TOKEN"))
    experiments = api.get_experiments(
        os.getenv("WORKSPACE"), project_name=os.getenv("PROJECT_NAME")
    )
    with open("outputs/job_status.json", "r") as fp:
        job_status = json.load(fp)

    rerun = []
    for exp in tqdm(experiments):
        exp_name = exp.get_name()
        metrics = pd.DataFrame(exp.get_metrics())
        metrics["metricValue"] = metrics["metricValue"].astype(float)
        if not Path(
            f"outputs/performance_info/{exp_name}_metrics.tsv"
        ).is_file():
            metrics.to_csv(
                f"outputs/performance_info/{exp_name}_metrics.tsv",
                sep="\t",
                index=False,
            )
        # check on R2
        not_there = "test/r2" not in metrics["metricName"].unique().tolist()
        not_calculated = np.isnan(
            metrics.loc[
                metrics["metricName"] == "test/r2", "metricValue"
            ].values
        )

        if len(exp.get_asset_list()) < 2:
            continue  # asset not saved correctly

        for deet in exp.get_asset_list():
            if "ckpt" not in deet["fileName"]:
                pass
            exp_key = (
                deet["curlDownload"]
                .split("experimentKey=")[-1]
                .split("&assetId")[0]
            )
            if not exp_key == job_status[exp_name]["comet_key"]:
                print(
                    f"Experiment key not matching. Something is wrong for {exp_name}..."
                    f"parsed from comet: {exp_key}"
                    f"parsed from slurm jobs: {job_status[exp_name]['comet_key']}"
                )
            curl_command = (
                deet["curlDownload"].split(" > ")[0]
                + " > outputs/model_registery/"
                + exp_name
                + ".ckpt"
            )
            ckpt_path = f"outputs/model_registery/{exp_name}.ckpt"
            if not Path(ckpt_path).is_file():
                c.run(curl_command)
            job_status[exp_name]["ckpt"] = ckpt_path

        if not_there or any(not_calculated):
            job_status[exp_name]["status"] = "miss_test/r2"
            rerun_command = f'python src/train.py -m  seed=1 experiment={exp_name} local=slurm_gpu train=False test=True  ckpt_path="{ckpt_path}" ++logger.comet.experiment_key={job_status[exp_name]["comet_key"]} ++hydra.launcher.mem_gb=6 ++hydra.launcher.cpus_per_task=5 ++hydra.launcher.timeout_min=30 ++hydra.sweep.subdir={job_status[exp_name]["comet_key"]}'
            rerun.append(rerun_command)

    # save job status
    with open("outputs/job_status.json", "w", encoding="utf-8") as f:
        json.dump(job_status, f, ensure_ascii=False, indent=4)
    print("& ".join(rerun))
