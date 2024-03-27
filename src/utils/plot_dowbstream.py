import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

path_data_small = (
    "outputs/ukbb_downstream_s/predict/simple_classifiers_sex.tsv"
)
path_data_medium = (
    "outputs/ukbb_downstream_m/predict/simple_classifiers_sex.tsv"
)

df_s = pd.read_csv(path_data_small, sep="\t", index_col=0)
df_s["model"] = "small"
df_m = pd.read_csv(path_data_medium, sep="\t", index_col=0)
df_m["model"] = "medium"
df = pd.concat([df_s, df_m], axis=0)

sns.set_theme(style="darkgrid")
f, ax = plt.subplots(figsize=(7, 5))
sns.despine(bottom=True, left=True)
for i, model in enumerate(["small", "medium"]):
    sns.pointplot(
        data=df[df["model"] == model],
        x="feature",
        y="score",
        hue="classifier",
        join=False,
        dodge=0.5 - 0.5 / 3,
        markers=["D", "o", "s", "v"],
        errorbar=None,
        ax=ax,
        palette=[sns.color_palette()[i]] * 4,
    )
ax.set_ylim(0.45, 1.0)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Feature")
ax.set_xticklabels(
    [
        "Connectome",
        "Average",
        "Standard\ndeviation",
        "Max",
        "Conv1d",
        "Average R$^2$",
        "R$^2$ map",
    ],
)
ax.hlines(0.5, -0.5, 6.5, linestyles="dashed", colors="black", label="Chance")
blue_patch = Patch(color=sns.color_palette()[0], label="Small model")
orange_patch = Patch(color=sns.color_palette()[1], label="Bigger model")
D_patch = Line2D(
    [0],
    [0],
    markerfacecolor="k",
    markeredgecolor="k",
    label="SVM",
    marker="D",
    ls="",
)
o_patch = Line2D(
    [0],
    [0],
    markerfacecolor="k",
    markeredgecolor="k",
    label="LogisticR",
    marker="o",
    ls="",
)
s_patch = Line2D(
    [0],
    [0],
    markerfacecolor="k",
    markeredgecolor="k",
    label="Ridge",
    marker="s",
    ls="",
)
v_patch = Line2D(
    [0],
    [0],
    markerfacecolor="k",
    markeredgecolor="k",
    label="MLP",
    marker="v",
    ls="",
)
chance = Line2D([0], [0], color="black", label="Chance", ls="--")

plt.legend(
    handles=[
        blue_patch,
        orange_patch,
        D_patch,
        o_patch,
        s_patch,
        v_patch,
        chance,
    ],
    loc="upper right",
)

ax.set_title("Down stream sex prediction accuracy")
plt.tight_layout()
plt.savefig("outputs/simple_classifiers_sex.png")
