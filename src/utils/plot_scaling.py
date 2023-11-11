"""
look through the `outputs/` directory, find instance of completed
training, and get the number of subjects used, mean R2 of test set,
plot the number of subjects (y-axis) against R2 (x axis)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PATH_PATTERN_SUCCESS_JOB = "multiruns/*/++experiment.scaling.n_sample=*,++experiment.scaling.segment=*,++random_state=*/r2_test.npy"

path_success_job = Path("outputs/scaling").glob(PATH_PATTERN_SUCCESS_JOB)
dict_n_sample_r2_test = pd.DataFrame()
for p in path_success_job:
    # parse the path and get number of subjects
    n_sample = int(p.parts[-2].split(",")[0].split("=")[-1])
    if n_sample == -1:
        n_sample = 25992
    # get random seed
    random_seed = int(p.parts[-2].split(",")[2].split("=")[-1])
    # load r2_test.npy
    r2_test = np.load(p)
    # get mean r2_test
    mean_r2_test = r2_test.mean()
    df = pd.DataFrame(
        [n_sample, random_seed, mean_r2_test],
        index=["n_sample", "random_seed", "mean_r2_test"],
    ).T
    dict_n_sample_r2_test = pd.concat([dict_n_sample_r2_test, df], axis=0)

dict_n_sample_r2_test.to_csv("outputs/scaling/_data.csv")
# plot
sns.lineplot(data=dict_n_sample_r2_test, x="n_sample", y="mean_r2_test")
plt.savefig("outputs/scaling/_plot.png")
plt.close()
