import json
import os
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
def migrate_flops_stats(c):
    load_dotenv()
    c.run(
        "rsync -vaRL outputs/performance_info/./* "
        f"{os.getenv('ELM_USER')}@{os.getenv('ELM_ADDRESS')}:"
        f"/home/{os.getenv('ELM_USER')}/simexp/{os.getenv('ELM_USER')}/"
        "ukbb_scaling_reports/data/ukbb_gcn_scaling/flops"
    )


@invoke.task
def local_job_status(c):
    log_dir = Path("outputs/autoreg/logs/train/multiruns")
    output_configs = log_dir.glob("**/.hydra/config.yaml")
    extra_time = 480

    if Path("outputs/local_job_status.json").is_file():
        with open("outputs/local_job_status.json") as f:
            job_status = json.load(f)
    else:
        job_status = {}

    for config_path in output_configs:
        exp_name = None
        date_completed = config_path.parents[2].name
        with open(config_path) as stream:
            config = yaml.safe_load(stream)
            if config.get("logger"):
                if config["logger"].get("comet"):
                    exp_name = config["logger"]["comet"]["experiment_name"]

        if not exp_name:
            continue

        log_path = config_path.parents[1] / "train.log"

        cur_status = {"time_out": 0, "complete": 0, "running": 0, "other": 1}
        with open(log_path, "r") as fp:
            lines = fp.readlines()
            for i, row in enumerate(lines):
                # check if string present on a current line
                if row.find("Job has timed out") != -1:
                    cur_status["time_out"] += 1
                elif row.find("Metric name is None!") != -1:
                    cur_status["complete"] += 1
                elif (
                    row.find("Starting training!") != -1
                    and i == len(lines) - 1
                ):
                    cur_status["running"] += 1
                else:
                    pass

        if 1 in [
            cur_status["complete"],
            cur_status["time_out"],
            cur_status["running"],
        ]:
            cur_status["other"] = 0

        for k, t in cur_status.items():
            if t == 1:
                status_str = k
        ckpt_path = list((log_path.parents[0] / "gcn").glob("**/*.ckpt"))
        if len(ckpt_path) > 0:
            ckpt_path = ckpt_path[0]
            comet_key = ckpt_path.parents[1].name
        else:
            ckpt_path, comet_key = None, None

        if exp_name not in job_status:
            job_status[exp_name] = {
                "status": status_str,
                "date": date_completed,
                "further_check": int(cur_status["other"] == 1),
                "ckpt": str(ckpt_path),
                "comet_key": comet_key,
            }
        elif datetime.strptime(
            date_completed, "%Y-%m-%d_%H-%M-%S"
        ) > datetime.strptime(
            job_status[exp_name]["date"], "%Y-%m-%d_%H-%M-%S"
        ):
            job_status[exp_name].update(
                {
                    "status": status_str,
                    "date": date_completed,
                    "further_check": int(cur_status["other"] == 1),
                    "ckpt": str(ckpt_path),
                    "comet_key": comet_key,
                }
            )
        else:
            pass

    rerun = []
    for exp in job_status:
        if job_status[exp]["status"] == "complete":
            print(f"Completed: {exp}.")
        elif job_status[exp]["further_check"] == 1:
            print(f"check {exp}: {job_status[exp]['ckpt']}")
        elif job_status[exp]["status"] == "running":
            pass
        else:
            ckpt_path = job_status[exp]["ckpt"].replace("=", "\\=")
            dataset = "" if "N-197" in exp else "data=ukbb_schaefer"
            command = f'python src/train.py -m {dataset} seed=1 experiment={exp} local=slurm_gpu ++logger.comet.experiment_key={job_status[exp]["comet_key"]} ++hydra.launcher.mem_gb=6 ++hydra.launcher.cpus_per_task=5 ++hydra.launcher.timeout_min={extra_time} ckpt_path="{ckpt_path}" ++hydra.sweep.subdir={job_status[exp]["comet_key"]}'
            job_status[exp]["rerun"] = command
            rerun.append(command)

    with open("outputs/local_job_status.json", "w", encoding="utf-8") as f:
        json.dump(job_status, f, ensure_ascii=False, indent=4)
    if rerun:
        print("rerun the following:")
        print(" &\n".join(rerun))


@invoke.task
def download_models(c):
    load_dotenv()
    api = API(os.getenv("COMET_API_TOKEN"))
    experiments = api.get_experiments(
        os.getenv("WORKSPACE"), project_name=os.getenv("PROJECT_NAME")
    )
    job_status = {}
    rerun = []
    all_exp_time = {}
    for exp in tqdm(experiments):
        exp_name = exp.get_name()
        exp_time = datetime.fromtimestamp(
            exp.get_parameters_summary()[0]["timestampCurrent"] / 1000
        )
        if exp_name in all_exp_time:
            if exp_time < all_exp_time[exp_name]:
                print("skip old log")
                continue

        all_exp_time[exp_name] = exp_time
        job_status[exp_name] = {"time": str(exp_time)}

        metrics = pd.DataFrame(exp.get_metrics())
        metrics["metricValue"] = metrics["metricValue"].astype(float)
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
        )  # there can be multiple values if we rerun the testing loop only

        if len(exp.get_asset_list()) < 2:
            print("skip. old experiment with assets missing.")
            continue  # asset not saved correctly

        for deet in exp.get_asset_list():
            job_status[exp_name]["comet_key"] = deet["experimentKey"]
            if "ckpt" not in deet["fileName"]:
                continue
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

        if not_there or all(not_calculated):
            job_status[exp_name]["status"] = "miss_test/r2"
            rerun_command = f'python src/train.py -m  seed=1 experiment={exp_name} local=slurm_gpu train=False test=True  ckpt_path="{ckpt_path}" ++logger.comet.experiment_key={job_status[exp_name]["comet_key"]} ++hydra.launcher.mem_gb=6 ++hydra.launcher.cpus_per_task=5 ++hydra.launcher.timeout_min=120 ++hydra.sweep.subdir={job_status[exp_name]["comet_key"]}'
            rerun.append(rerun_command)
        else:
            job_status[exp_name]["status"] = "downloaded"

    # save job status
    with open("outputs/job_status.json", "w", encoding="utf-8") as f:
        json.dump(job_status, f, ensure_ascii=False, indent=4)
    print("& ".join(rerun))
