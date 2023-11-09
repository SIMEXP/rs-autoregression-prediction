"""
look through the `outputs/` directory, find instance of completed
training, and get the number of subjects used, mean R2 of test set,
plot the number of subjects (y-axis) against R2 (x axis)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PATH_PATTERN_SUCCESS_JOB = "multiruns/*/++experiment.scaling.n_sample=*,++experiment.scaling.segment=*/r2_test.npy"

path_success_job = Path("outputs/scaling").glob(PATH_PATTERN_SUCCESS_JOB)
dict_n_sample_r2_test = {}
for p in path_success_job:
    # parse the path and get number of subjects
    n_sample = int(p.parts[-2].split(",")[0].split("=")[-1])
    if n_sample == -1:
        n_sample = 25992
    # load r2_test.npy
    r2_test = np.load(p)
    # get mean r2_test
    mean_r2_test = r2_test.mean()
    # save to dict with key = n_sample, value = mean_r2_test
    dict_n_sample_r2_test[n_sample] = mean_r2_test
# sort the dictionary by key
sorted_n_sample = sorted(dict_n_sample_r2_test.keys())
dict_n_sample_r2_test = {k: dict_n_sample_r2_test[k] for k in sorted_n_sample}
# plot
plt.plot(
    list(dict_n_sample_r2_test.keys()), list(dict_n_sample_r2_test.values())
)
plt.xlabel("Number of subjects")
plt.ylabel("Mean R2 of test set")
plt.title("UK Biobank")
plt.savefig("outputs/scaling/_plot.png")
plt.close()
