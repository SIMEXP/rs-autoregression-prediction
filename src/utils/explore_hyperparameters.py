import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd

output_dirs = Path(
    "outputs/autoreg/train/multiruns/nembed-197_hyperparameters"
).glob("**/train.log")
data = []


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return itertools.chain([first], iterable)


for p in output_dirs:
    experiment = {
        f.groups()[0]: float(f.groups()[1])
        for f in re.finditer(r"model\.([a-z_]*)=([\d\.e?-]*)", p.parent.name)
    }
    experiment["mean_r2_val"] = np.nan
    experiment["runtime"] = np.nan
    if (p.parent / "model.pkl").exists():
        with open(p, "r") as log:
            report = log.read()
        mean_r2_val = re.search(r"Mean r2 val: ([\-\.\d]*)", report).groups()[
            0
        ]
        starttime = re.search(r"\[([\d\-\s:,]*)\].*Process ID", report).group(
            1
        )
        endtime = re.search(r"\[([\d\-\s:,]*)\].*model trained", report).group(
            1
        )
        starttime = pd.to_datetime(starttime)
        endtime = pd.to_datetime(endtime)
        runtime = endtime - starttime
        experiment["mean_r2_val"] = mean_r2_val
        experiment["runtime"] = runtime.total_seconds() / 60
    data.append(experiment)

data = pd.DataFrame(data)
data = data.sort_values("mean_r2_val", ascending=False)
data.to_csv("_explore_hyperparameters.tsv", sep="\t", index=False)
