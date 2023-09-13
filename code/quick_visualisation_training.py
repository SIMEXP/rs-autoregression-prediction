import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from seaborn import boxplot, lineplot
from statsmodels.formula.api import ols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", "-m", type=Path, help="model output directory"
    )
    parser.add_argument(
        "--horizon_dir", "-p", type=Path, help="Path to horizon prediction."
    )
    args = parser.parse_args()

    (args.model_dir / "figures").mkdir(exist_ok=True)
    (args.horizon_dir / "figures").mkdir(exist_ok=True)

    model_name = args.model_dir.name

    # visualise training loss
    training_losses = np.load(
        args.model_dir / "training_losses.npy", allow_pickle=True
    ).tolist()
    training_losses = pd.DataFrame(training_losses)

    plt.figure()
    g = lineplot(data=training_losses)
    g.set_xlabel("Epoc")
    g.set_ylabel("Loss (MSE)")
    for fold in ["tng", "val", "test"]:
        r2 = np.load(args.model_dir / f"r2_{fold}.npy", allow_pickle=True)
        print(f"r2 {fold}: {np.mean(r2)}")
    plt.savefig(args.model_dir / "figures/training_losses.png")

    # visualise r2mean by phenotype information and sites
    r2mean = pd.read_csv(
        args.horizon_dir / f"{model_name}_horizon-1.tsv", index_col=0, sep="\t"
    )
    r2mean = r2mean[(r2mean.r2mean > -1e16)]  # remove outliers

    plt.figure()
    g = boxplot(x="site", y="r2mean", hue="diagnosis", data=r2mean)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.savefig(args.horizon_dir / "figures/diagnosis_by_sites.png")

    plt.figure()
    g = boxplot(x="site", y="r2mean", hue="sex", data=r2mean)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.savefig(args.horizon_dir / "figures/sex_by_sites.png")
