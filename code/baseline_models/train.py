import numpy as np
from sklearn.metrics import r2_score
from src.data.load_data import Dataset
from src.models.models import Chebnet, LRMultivariate, LRUnivariate
from src.tools import iter_fun, string_to_list
from torch import optim
from torch.cuda import is_available as cuda_is_available
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

NUM_WORKERS = 2
DEVICE = "cuda:0"


def train_backprop(model, X_tng, Y_tng, X_val, Y_val, params, verbose=1):
    """Backprop training of pytorch models for binary classification.
    Returns trained model, losses and checkpoints.
    """
    tng_dataset = Dataset(X_tng, Y_tng)
    val_dataset = Dataset(X_val, Y_val)
    tng_dataloader = DataLoader(
        tng_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    if not cuda_is_available():
        device = "cpu"
        print("CUDA not available, running on CPU.")
    else:
        device = params["torch_device"] if "torch_device" in params else DEVICE
        if verbose:
            print(f"Using device {device}")

    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=params["lr_patience"],
        threshold=params["lr_thres"],
    )
    loss_function = BCELoss().to(device)
    losses = {"tng": [], "val": []}

    if "checkpoints" in params:
        checkpoints = string_to_list(params["checkpoints"])
    else:
        checkpoints = []
    checkpoint_scores = []

    # training loop
    for epoch in iter_fun(range(params["nb_epochs"]), verbose):
        model.train()
        mean_loss_tng = 0.0
        is_checkpoint = epoch in checkpoints
        all_preds_tng = []
        all_labels_tng = []
        all_preds_val = []
        all_labels_val = []
        for sampled_batch in tng_dataloader:
            optimizer.zero_grad()
            inputs = sampled_batch["input"].to(device)
            labels = sampled_batch["label"].to(device)
            preds = model(inputs)
            loss = loss_function(preds, labels)
            mean_loss_tng += loss.item()
            loss.backward()
            optimizer.step()
            if is_checkpoint:
                all_preds_tng.append(preds.detach().cpu().numpy())
                all_labels_tng.append(labels.detach().cpu().numpy())
        mean_loss_tng = mean_loss_tng / len(tng_dataloader)
        scheduler.step(mean_loss_tng)
        losses["tng"].append(mean_loss_tng)

        # compute validation loss
        model.eval()
        mean_loss_val = 0.0
        for sampled_batch in val_dataloader:
            inputs = sampled_batch["input"].to(device)
            labels = sampled_batch["label"].to(device)
            preds = model(inputs)
            loss = loss_function(preds, labels)
            mean_loss_val += loss.item()
            if is_checkpoint:
                all_preds_val.append(preds.detach().cpu().numpy())
                all_labels_val.append(labels.detach().cpu().numpy())
        mean_loss_val = mean_loss_val / len(val_dataloader)
        losses["val"].append(mean_loss_val)

        if verbose > 1:
            print(
                "epoch",
                epoch,
                "tng loss",
                mean_loss_tng,
                "val loss",
                mean_loss_val,
            )

        # add checkpoint
        if is_checkpoint:
            acc_tng = (
                np.concatenate(all_preds_tng, axis=0)
                == np.concatenate(all_labels_tng, axis=0)
            ).float()

            acc_val = (
                np.concatenate(all_preds_val, axis=0)
                == np.concatenate(all_labels_val, axis=0)
            ).float()
            score_dict = {}
            score_dict["epoch"] = epoch
            score_dict["acc_mean_tng"] = acc_tng.mean()
            score_dict["acc_std_tng"] = acc_tng.std()
            score_dict["acc_mean_val"] = acc_val.mean()
            score_dict["acc_std_val"] = acc_val.std()
            score_dict["loss_tng"] = mean_loss_tng
            score_dict["loss_val"] = mean_loss_val
            checkpoint_scores.append(score_dict)

    if verbose:
        print("model trained")

    return model, losses, checkpoint_scores
