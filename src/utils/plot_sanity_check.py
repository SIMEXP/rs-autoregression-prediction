# plot the model trained on the full UK Biobank dataset for sanity check
# plot prediction vs. label (ground truth)
# plot the power spectrum of the same segment
# average power spectrum of all subjects
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from src.data.load_data import load_data

path = Path("outputs/scaling/multiruns/2023-11-12_22-55-57/")
model = '++experiment.scaling.n_sample=-1,++experiment.scaling.segment=1,++model.FK="32,6,32,6,32,6,16,6,16,6",++model.M="32,16,8,1",++model.dropout=0.1,++random_state=1'
seq = 16

labels_val = np.load(path / model / "labels_val.npy")
pred_val = np.load(path / model / "pred_val.npy")

with open(path / model / "train_test_split.json", "r") as f:
    train_test_split = json.load(f)
ts_val = load_data(
    "inputs/connectomes/ukbb.h5", train_test_split["val"][0], dtype="data"
)
length_1st_suj = ts_val[0].shape[0] - seq

example_original = zscore(labels_val[:length_1st_suj, 60])
example_pred = zscore(pred_val[:length_1st_suj, 60])

plt.figure()
plt.plot(
    (np.arange(len(example_original)) + seq) * (0.735 * 4),
    example_original,
    color="k",
    alpha=0.2,
    label="original",
)
plt.plot(
    (np.arange(len(example_original)) + seq) * (0.735 * 4),
    example_pred,
    label="pred lag 1",
)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("z-scored BOLD signal")
plt.title("Prediction vs. True signal\n(one subject; ROI 59 from MIST 197)")
plt.savefig("outputs/scaling/timeseres_pred_vs_original.png")
plt.close()

# calculate power spectrum of the same segment
example_original_power = np.abs(np.fft.fft(example_original, axis=0))
example_pred_power = np.abs(np.fft.fft(example_pred))
freq = np.fft.fftfreq(example_pred.shape[0], d=1 / (0.735 * 4))

plt.figure()
plt.plot(
    freq[:53],
    example_original_power[:53],
    color="k",
    alpha=0.2,
    label="original",
)
plt.plot(freq[:53], example_pred_power[:53], label="pred lag 1")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.title(
    "Prediction vs. True signal Power Spectrum\n(one subject; ROI 59 from MIST 197)"
)
plt.savefig("outputs/scaling/power_pred_vs_original.png")
plt.close()
