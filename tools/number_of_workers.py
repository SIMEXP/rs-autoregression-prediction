import multiprocessing as mp
from time import time

import h5py
from fmri_autoreg.data.load_data import Dataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

proportion_sample = 1
tng_data_h5 = (
    "outputs/sample_for_pretraining/seed-42/sample_seed-42_data-train.h5"
)
IS_GPU = False

with open("outputs/number_of_workers.txt", "w") as f:
    f.write("batch_size\tn_embed\tnum_workers\tepoch_second\n")

for n_embed in [64, 197, 444]:
    if proportion_sample != 1:
        with h5py.File(tng_data_h5, "r") as f:
            tng_length = f[f"n_embed-{n_embed}"]["train"]["input"].shape[0]
        tng_index = list(range(int(tng_length * proportion_sample)))
        tng_dataset = Subset(
            Dataset(
                tng_data_h5, n_embed=f"n_embed-{n_embed}", set_type="train"
            ),
            tng_index,
        )
    else:
        tng_dataset = Dataset(
            tng_data_h5, n_embed=f"n_embed-{n_embed}", set_type="train"
        )
    for batch_size in [512]:
        for num_workers in range(4, 34, 2):
            train_loader = DataLoader(
                tng_dataset,
                shuffle=True,
                num_workers=num_workers,
                batch_size=batch_size,
                pin_memory=IS_GPU,
            )
            start = time()
            for _ in tqdm(
                range(1, 3),
                desc=f"batch_size={batch_size}; n_embed={n_embed}; Number of workers: {num_workers}",
            ):
                for _, _ in enumerate(train_loader, 0):
                    pass
            end = time()
            taken = (end - start) / 2
            with open("outputs/number_of_workers.txt", "a") as f:
                f.write(f"{batch_size}\t{n_embed}\t{num_workers}\t{taken}\n")
