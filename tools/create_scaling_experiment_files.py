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

import itertools

GCL_options = [3, 6, 9]
# F_options = [8, 16, 32, 64, 128]
F_options = [8, 16, 32]
K_options = [3, 5, 10]
FCL_options = [1, 3, 5]
M_options = [8, 16, 32]

# N_PARCEL_options = [64, 197, 444]
N_PARCEL_options = [197]
WINDOW_options = [16]
# WINDOW_options = [16,8]


if __name__ == "__main__":
    all_combbinations = list(
        itertools.product(
            *[
                GCL_options,
                F_options,
                K_options,
                FCL_options,
                M_options,
                N_PARCEL_options,
                WINDOW_options,
            ]
        )
    )

    n_exp = 0
    for current_set in all_combbinations:
        GCL, F, K, FCL, M, N_PARCEL, WINDOW = current_set
        if F < N_PARCEL:
            n_exp += 1
    #         replace = {
    #             "REPLACE_FK": f"{F},{K}," * GCL [:-1],
    #             "REPLACE_M": f"{M}," * FCL + "1",
    #             "EXP_NAME": f"N-{N_PARCEL}_W-{WINDOW}_GCL-{GCL}_F-{F}_K-{F}_FCL-{FCL}_M-{M}"
    #         }

    #         output_path = f"configs/experiment/{replace['EXP_NAME']}.yaml"

    #         config = TEMPLATE.copy()

    #         for k in replace:
    #             config = config.replace(k, replace[k])

    #         with open(output_path, "w") as f:
    #             f.write(config)
    print(f"Generate {n_exp} experiments.")
