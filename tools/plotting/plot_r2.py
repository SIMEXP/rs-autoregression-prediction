from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.data.load_data import load_data, load_h5_data_path, load_phenotype

phenotype_file = "inputs/connectomes/sourcedata/ukbb/ukbb_pheno.tsv"
data_file = "outputs/ukbb_downstream_s/extract/feature_horizon-1.h5"
output_dir = "outputs/ukbb_downstream_s/extract/figures"

phenotype_file = Path(phenotype_file)
data_file = Path(data_file)
output_dir = Path(output_dir)

# combine average t + 1 R2 and phenotype data
df_phenotype = load_phenotype(phenotype_file)

dset_path = load_h5_data_path(
    data_file,
    "r2",
    shuffle=False,
)
data = load_data(data_file, dset_path, dtype="data")
data = np.concatenate(data).squeeze()
data = data.mean(axis=1).reshape(-1, 1)
participant_id = [
    p.split("/")[-1].split("sub-")[-1].split("_")[0] for p in dset_path
]
r2_avg = pd.DataFrame(data, columns=["r2mean"], index=participant_id)
df_for_stats = df_phenotype.join(r2_avg, how="inner")
df_for_stats.to_csv(output_dir / "r2_phenotype.tsv", sep="\t")
# quick plotting
df_for_stats = df_for_stats[(df_for_stats.r2mean > 0)]  # remove outliers

if "diagnosis" in df_for_stats.columns:
    plt.figure()
    g = sns.boxplot(x="site", y="r2mean", hue="diagnosis", data=df_for_stats)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    plt.savefig(output_dir / "diagnosis_by_sites.png")

plt.figure()
g = sns.boxplot(x="site", y="r2mean", hue="sex", data=df_for_stats)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.savefig(output_dir / "sex_by_sites.png")

plt.figure()
g = sns.lmplot(x="age", y="r2mean", hue="site", data=df_for_stats)
plt.savefig(output_dir / "age_by_sites.png")
