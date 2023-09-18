import numpy as np
import pandas as pd
import statsmodels.api as sm
from seaborn import boxplot, lineplot
from statsmodels.formula.api import ols

PRETRAIN = "outputs/prototype_train_and_test_within-sites_original-tr"
HORIZON = "outputs/prototype_predict_horizon_within-sites_original-tr"

if __name__ == "__main__":
    model_name = PRETRAIN.split("/")[-1]

    # visualise training loss
    training_losses = np.load(
        PRETRAIN + "/training_losses.npy", allow_pickle=True
    ).tolist()
    training_losses = pd.DataFrame(training_losses)
    g = lineplot(data=training_losses)
    g.set_xlabel("Epoc")
    g.set_ylabel("Loss (MSE)")
    for fold in ["tng", "val", "test"]:
        r2 = np.load(f"{PRETRAIN}/r2_{fold}.npy", allow_pickle=True)
        print(f"r2 {fold}: {np.mean(r2)}")
    g.savefig(f"{PRETRAIN}/figures/training_losses.png")

    # visualise r2mean by phenotype information and sites
    r2mean = pd.read_csv(f"{HORIZON}/{model_name}.tsv", index_col=0, sep="\t")
    r2mean = r2mean[(r2mean.r2mean > -1e16)]  # remove outliers

    g = boxplot(x="site", y="r2mean", hue="diagnosis", data=r2mean)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.savefig(f"{HORIZON}/figures/diagnosis_by_sites.png")

    g = boxplot(x="site", y="r2mean", hue="sex", data=r2mean)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.savefig(f"{HORIZON}/figures/sex_by_sites.png")
