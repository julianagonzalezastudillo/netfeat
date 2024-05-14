import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
from collections import Counter

from plottools.plot_positions import channel_pos
from plottools.plot_tools import colorbar
from config import load_config, ConfigPath, DATASETS


# Load params
params, paradigm = load_config()
pipeline = "RG+SVM"

# create dict with all the possible channels in all datasets
ch_keys = np.unique(sum(params["ch_names"].values(), []))
ch_dict = dict.fromkeys(ch_keys, 0)

# in order to normalize, we need to know
# how many times each channel could be selected: ch*dt_sub*dt_sessions*cv
ch_list = sum(
    [
        params["ch_names"][dt_.code] * len(dt_.subject_list) * dt_.n_sessions * 5
        for dt_ in DATASETS
    ],
    [],
)
ch_counts = Counter(ch_list)

for dt in DATASETS:
    # load selected features file
    filename = (
        ConfigPath.RES_DIR / f"select_features/select_features_{dt.code}_{pipeline}.csv"
    )
    fs = pd.read_csv(filename)

    # all selected channels
    ch = [c for ch_cv in fs["ch"] for c in ch_cv.strip("']['").split("' '")]

    # count channels
    ch_dict.update({ch_i: ch_dict[ch_i] + ch.count(ch_i) for ch_i in np.unique(ch)})

# normalize by times it could have been selected == occurrences
ch_norm = np.array(
    [ch_dict[ch] / ch_counts[ch] if ch_counts[ch] != 0 else 0 for ch in ch_keys]
)
ch_size_norm_to_plot = ch_norm / max(abs(ch_norm))  # for normalize size

# get electrodes positions
ch_pos = channel_pos(ch_keys, dimension="2d")

# Create 2D figure
fig = plt.figure(figsize=(7, 5), dpi=300)
colors = [[0, "#e1e8ed"], [1.0, "#ff843c"]]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
im = plt.scatter(
    ch_pos[:, 0],
    ch_pos[:, 1],
    s=(abs(ch_size_norm_to_plot) * fig.dpi / 6) ** 2,
    c=ch_norm,
    marker=".",
    cmap=cmap,
    alpha=1,
    linewidths=0,
    edgecolors="k",
)

for i, ch in zip(np.arange(len(ch_keys)), ch_keys):
    plt.text(
        ch_pos[i, 0],
        ch_pos[i, 1],
        ch,
        fontsize=6,
        horizontalalignment="center",
        verticalalignment="center",
        c="k",
        fontname="Arial",
    )

plt.gca().set_aspect("equal", adjustable="box")
plt.axis("off")
# plt.title("RG occurrences")
colorbar(fig, im)


plt.show()
fig_name = (
    ConfigPath.RES_DIR
    / "riemannian_features/plot/riemannian_occurrences_mean_across_sub.png"
)
# fig.savefig(fig_name, transparent=True)
