TEMPLATE = """# @package _global_

# to execute this experiment run:
# python src/train.py experiment=example

defaults:
  - override /model: chebnet_default
  - override /callbacks: scaling
  - override /trainer: gpu
  - override /logger: comet

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

tags: [ukbb, gcn, dev]

train: true

test: true

proportion_sample: 1.0

trainer:
  min_epochs: 1
  max_epochs: 20
  limit_train_batches: ${proportion_sample}
  limit_val_batches: ${proportion_sample}

model:
  net:
    _target_: src.models.components.chebnet.Chebnet
    n_emb: ${data.atlas[1]}
    seq_len: ${data.timeseries_window_stride_lag[0]}
    FK: REPLACE_FK
    M: RPLACE_M
    FC_type: nonshared_uni
    dropout: 0
    bn_momentum: 0.1

data:
  atlas: [MIST, N_PARCEL]
  timeseries_window_stride_lag: [WINDOW, 1, 1]
  num_workers: 8
  pin_memory: true

logger:
  comet:
    experiment_name: EXP_NAME
"""

import argparse
import itertools
from pathlib import Path

GCL_options = [3, 6, 9]
F_options = [8, 16, 32]
K_options = [3, 5, 10]
FCL_options = [1, 3, 5]
M_options = [8, 16, 32]

N_PARCEL_options = [64, 197, 444]
WINDOW_options = [16, 8]


def create_template(
    GCL, F, K, FCL, M, n_parcels, time_window, output_dir, dry_run
):
    replace = {
        "REPLACE_FK": str(f"{F},{K}," * GCL)[:-1],
        "REPLACE_M": f"{M}," * FCL + "1",
        "EXP_NAME": f"N-{n_parcels}_W-{time_window}_GCL-{GCL}_F-{F}_K-{K}_FCL-{FCL}_M-{M}",
    }

    output_path = output_dir / f"{replace['EXP_NAME']}.yaml"
    if (
        Path("configs/experiment/completed") / f"{replace['EXP_NAME']}.yaml"
    ).is_file():
        print(f"Already completed: {replace['EXP_NAME']}.yaml")
        return 0

    if not dry_run:
        for k in replace:
            config = TEMPLATE.replace(k, replace[k])

        with open(output_path, "w") as f:
            f.write(config)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to save the configs.",
    )
    parser.add_argument(
        "--GCL", type=int, nargs="+", help="Number of ChebNet layers."
    )
    parser.add_argument(
        "--F", type=int, nargs="+", help="Number of filters in ChebNet layers."
    )
    parser.add_argument(
        "--K",
        type=int,
        nargs="+",
        help="Number of polynomials in ChebNet layers.",
    )
    parser.add_argument(
        "--FCL", type=int, nargs="+", help="Number of fully connected layers."
    )
    parser.add_argument(
        "--M",
        type=int,
        nargs="+",
        help="Number of nodes in fully connected layers.",
    )
    parser.add_argument(
        "--n-parcels",
        type=int,
        nargs="+",
        help="Number of parcels.",
        choices=N_PARCEL_options,
    )
    parser.add_argument(
        "--n-timepoints",
        type=int,
        nargs="+",
        help="Number of timepoints in a window.",
        choices=WINDOW_options,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't generate actual files.",
    )
    args = parser.parse_args()
    print(args)

    output_dir = args.output_dir

    all_combbinations = list(
        itertools.product(
            *[
                args.GCL,
                args.F,
                args.K,
                args.FCL,
                args.M,
                args.n_parcels,
                args.n_timepoints,
            ]
        )
    )

    n_exp = 0
    for current_set in all_combbinations:
        GCL, F, K, FCL, M, N_PARCEL, WINDOW = current_set
        if F < N_PARCEL:
            n_exp += create_template(
                GCL, F, K, FCL, M, N_PARCEL, WINDOW, output_dir, args.dry_run
            )
    print(f"Generate {n_exp}/{len(all_combbinations)} experiments.")


if __name__ == "__main__":
    main()
