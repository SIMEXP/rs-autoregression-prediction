import csv
from pathlib import Path
import argparse
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
from src.models.predict_model import predict_horizon
from src.tools import check_path
from src.models.train_model import train


HORIZON = 6


def main():
    """Train model using parameters dict and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=Path, help="output directory")
    parser.add_argument("--param", "-p", type=Path, help="Parameters : path to json file or dict")
    parser.add_argument("-n", type=int, help="Number of subjects to use.")
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        help="Verbosity level, 0 to 2. Default is 1.",
    )
    args = parser.parse_args()

    params = load_params(args.param)
    batch_size = params["batch_size"] if "batch_size" in params else 100
    standardize = params["standardize"] if "standardize" in params else False
    compute_edge_index = params["model"] == "Chebnet"
    thres = params["edge_index_thres"] if compute_edge_index else None
    output_dir = args.output_dir
    output_dir = check_path(output_dir)
    os.makedirs(output_dir)

    fieldnames = list(params.keys()) + [
        "r2_mean_tng",
        "r2_std_tng",
        "r2_mean_test",
        "r2_std_test",
        "sub",
        "n_vol",
    ]

    if not os.path.exists(args.output):
        with open(args.output, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()

    tng_data = load_data(
        params["tng_data_file"], params["tng_task_filter"], standardize
        )
    val_data = load_data(
        params["val_data_file"], params["val_task_filter"], standardize
        )

    tng_data_subjects = tng_data[: args.n]
    n_vol = sum([d.shape[0] for d in tng_data_subjects])

    X_tng, Y_tng, X_val, Y_val, edge_index = make_input_labels(
        tng_data_subjects,
        val_data,
        params["seq_length"],
        params["time_stride"],
        params["lag"],
        compute_edge_index,
        thres,
        )
    print(f"{args.n} runs, {n_vol} volumes, training shape: {X_tng.shape}")
    trial_data = (X_tng, Y_tng, X_val, Y_val, edge_index)
    del tng_data
    del val_data
    model, r2_tng, r2_val, Z_tng, Y_tng, Z_val, Y_val, _, _ = train(
        params, trial_data, verbose=args.verbose
        )

    params["r2_mean_tng"] = r2_tng.mean()
    params["r2_std_tng"] = r2_tng.std()
    params["n_vol"] = n_vol

    # predict t + horizon
    r2_test, Z_test, Y_test = predict_horizon(
        model,
        params["seq_length"],
        HORIZON,
        params["val_data_file"],
        params["val_task_filter"],
        batch_size,
    )[0]

    for i in range(HORIZON):
        params["lag"] = i + 1
        params["r2_mean_test"] = r2_test[i].mean()
        params["r2_std_test"] = r2_test[i].std()

        with open(args.output, "a") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writerow(params)
    print("done.")



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

    np.save(os.path.join(output_dir, "pred_test.npy"), Z_test)
    np.save(os.path.join(output_dir, "labels_test.npy"), Y_test)


if __name__ == "__main__":
    main()
