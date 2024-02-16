import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

path_data_small = "outputs/predict/multiruns/2024-02-15_17-39-28/0/simple_classifiers_sex.tsv"
path_data_medium = "outputs/predict/multiruns/2024-02-15_17-39-28/1/simple_classifiers_sex.tsv"

df_s = pd.read_csv(path_data_small, sep="\t", index_col=0)
df_m = pd.read_csv(path_data_medium, sep="\t", index_col=0)
df = pd.concat([df_s, df_m], axis=0)

plt.figure()
sns.set_theme(style="whitegrid")
f, axs = plt.subplots(nrows=2, figsize=(9, 13))
sns.despine(bottom=True, left=True)
for model, ax in zip(["small", "medium"], axs):
    sns.pointplot(
        data=df[df["model"] == model],
        x="feature",
        y="score",
        hue="classifier",
        join=False,
        dodge=0.4 - 0.4 / 3,
        markers="d",
        scale=0.75,
        errorbar=None,
        ax=ax,
    )
    sns.move_legend(
        ax,
        loc="upper right",
        ncol=1,
        frameon=True,
        columnspacing=1,
        handletextpad=0,
    )
    ax.set_ylim(0.4, 1.0)
    ax.hlines(0.5, -0.5, 5.5, linestyles="dashed", colors="black")
    ax.set_title(f"Down stream sex prediction accuracy: {model} model")
plt.tight_layout()
plt.savefig("simple_classifiers_sex.png")
