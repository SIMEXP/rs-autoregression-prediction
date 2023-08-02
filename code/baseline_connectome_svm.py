"""
Execute at the root of the repo, not in the code directory.
"""
import h5py
import numpy as np
import pandas as pd
from giga_companion.load_data import load_data, load_h5_data_path
from nilearn.connectome import ConnectivityMeasure
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

data_file = "inputs/connectomes/processed_connectomes.h5"


if __name__ == "__main__":
    # ABIDE 1
    abide1 = load_h5_data_path(
        data_file,
        "abide1.*/*/sub-.*desc-197.*timeseries",
        shuffle=True,
    )
    # ABIDE 2
    abide2 = load_h5_data_path(
        data_file,
        "abide2.*/*/sub-.*desc-197.*timeseries",
        shuffle=True,
    )

    cm = ConnectivityMeasure(kind="correlation", vectorize=True)
    clf = LinearSVC(C=100, penalty="l2", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # get data
    tng_data = cm.fit_transform(
        [load_data(data_file, d, dtype="data")[0] for d in abide1]
    )
    tng_label = [load_data(data_file, d, dtype="diagnosis")[0] for d in abide1]
    tng_sites = [d.split("/")[1] for d in abide1]

    test_data = cm.fit_transform(
        [load_data(data_file, d, dtype="data")[0] for d in abide2]
    )
    test_label = [
        load_data(data_file, d, dtype="diagnosis")[0] for d in abide2
    ]

    # cross validation on abide 1 to understand the performance
    val_acc = []
    for tng_idx, val_idx in cv.split(tng_data, tng_sites):
        tng_conn = tng_data[tng_idx]
        val_conn = tng_data[val_idx]
        tng_dx = [tng_label[i] for i in tng_idx]
        val_dx = [tng_label[i] for i in val_idx]

        clf.fit(tng_conn, tng_dx)
        tng_pred = clf.predict(tng_conn)
        val_pred = clf.predict(val_conn)

        val_acc.append(accuracy_score(val_dx, val_pred))
    print(
        f"5-fold cv to show the accuracy on abide 1: mean={np.mean(val_acc):.3f}, std={np.std(val_acc):.3f}"
    )

    # train on ABIDE 1, test on ABIDE 2
    clf.fit(tng_data, tng_label)
    test_pred = clf.predict(test_data)
    test_acc = accuracy_score(test_label, test_pred)
    print(f"Train on ABIDE 1, Test on ABIDE 2 accuracy: {test_acc:.3f}")
