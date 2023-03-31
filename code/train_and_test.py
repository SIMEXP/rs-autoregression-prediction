import numpy as np
import os
import pickle as pk
from math import ceil
from sklearn.metrics import r2_score
from src.data.load_data import (
    load_params,
    load_data,
    make_input_labels,
    make_seq,
)
from src.tools import check_path
from src.models.train_model import train

mist_197 = "inputs/atlas/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_atlas-MIST_res-dataset_desc-197_dseg.nii.gz"

params = load_params("code/parameters.json")
standardize = params["standardize"] if "standardize" in params else False
compute_edge_index = params["model"] == "Chebnet"
thres = params["edge_index_thres"] if compute_edge_index else None
output_dir = "outputs/smoke_test"
output_dir = check_path(output_dir)
os.makedirs(output_dir)

tng_data = load_data(
    params["tng_data_file"], params["tng_task_filter"], standardize
    )
val_data = load_data(
    params["val_data_file"], params["val_task_filter"], standardize
    )
test_data = load_data(
    params["test_data_file"], params["test_task_filter"], standardize
    )

data = make_input_labels(
    tng_data,
    val_data,
    params["seq_length"],
    params["time_stride"],
    params["lag"],
    compute_edge_index,
    thres,
    )
del tng_data
del val_data

model, r2_tng, r2_val, Z_tng, Y_tng, Z_val, Y_val, _, _ = train(
    params, data, verbose=1
    )
X_test, Y_test = make_seq(
    test_data, params["seq_length"], params["time_stride"], params["lag"]
)
del test_data
batch_size = params["batch_size"]
Z_test = np.concatenate(
    [
        model.predict(x)
        for x in np.array_split(X_test, ceil(X_test.shape[0] / batch_size))
    ]
)
r2_test = r2_score(Y_test, Z_test, multioutput="raw_values")

model = model.to("cpu")
with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
    pk.dump(model, f)

np.save(os.path.join(output_dir, "r2_tng.npy"), r2_tng)
np.save(os.path.join(output_dir, "r2_val.npy"), r2_val)
np.save(os.path.join(output_dir, "r2_test.npy"), r2_test)
np.save(os.path.join(output_dir, "pred_tng.npy"), Z_tng)
np.save(os.path.join(output_dir, "labels_tng.npy"), Y_tng)
np.save(os.path.join(output_dir, "pred_val.npy"), Z_val)
np.save(os.path.join(output_dir, "labels_val.npy"), Y_val)
np.save(os.path.join(output_dir, "pred_val.npy"), Z_test)
np.save(os.path.join(output_dir, "labels_val.npy"), Y_test)


# just testing the plotting
from nilearn.maskers import NiftiLabelsMasker

masker = NiftiLabelsMasker(mist_197).fit()

unmasked = masker.inverse_transform(r2_val)

html_view = view_img(unmasked)
html_view.save_as_html(os.path.join(output_dir,'r2_val.html'))
