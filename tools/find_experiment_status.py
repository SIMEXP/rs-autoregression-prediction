import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

EXTRA_TIME = 120


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Path of multiruns log directory.",
    )
    parser.add_argument(
        "--extra-time",
        type=int,
        default=EXTRA_TIME,
        help="Extra time for incomplete jobs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't generate actual files.",
    )
    args = parser.parse_args()
    print(args)

    output_configs = args.log_dir.glob("**/.hydra/config.yaml")
    extra_time = args.extra_time

    if Path("job_status.json").is_file():
        with open("job_status.json") as f:
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
            command = f'python src/train.py -m  seed=1 experiment={exp} local=slurm_gpu ++logger.comet.experiment_key={job_status[exp]["comet_key"]} ++hydra.launcher.mem_gb=6 ++hydra.launcher.cpus_per_task=5 ++hydra.launcher.timeout_min={extra_time} ckpt_path="{ckpt_path}" ++hydra.sweep.subdir={job_status[exp]["comet_key"]}'
            job_status[exp]["rerun"] = command
            rerun.append(command)

    with open("outputs/job_status.json", "w", encoding="utf-8") as f:
        json.dump(job_status, f, ensure_ascii=False, indent=4)
    if rerun:
        print("rerun the following:")
        print(" &\n".join(rerun))


if __name__ == "__main__":
    main()
