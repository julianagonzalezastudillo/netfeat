import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
from collections import Counter

from plottools.plot_positions import channel_pos
from plottools.plot_tools import colorbar, save_mat_file
from config import load_config, ConfigPath, DATASETS, EXCLUDE_CHANNELS


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
ch_keys = ch_keys[~np.isin(ch_keys, EXCLUDE_CHANNELS)]

# normalize by times it could have been selected == occurrences
ch_norm = np.array(
    [ch_dict[ch] / ch_counts[ch] if ch_counts[ch] != 0 else 0 for ch in ch_keys]
)
ch_size_norm_to_plot = ch_norm / max(abs(ch_norm))  # for normalize size

# get electrodes positions
ch_pos = channel_pos(ch_keys, dimension="2d")

# Create 2D figure
fig = plt.figure(figsize=(7, 5), dpi=300)
# Define colors
colors = [
    [0.0, "#ffffe0"],
    [0.25, "#eceaaf"],
    [0.5, "#8cc379"],
    [0.75, "#269042"],
    [1.0, "#05550c"],
]
# f7fcb9
# addd8e
# 31a354
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

# Get 3D layout and save
norm = plt.Normalize(vmin=0, vmax=max(abs(ch_norm)))
rgb_values = cmap(norm(ch_norm))

save_mat_file(
    ch_norm, rgb_values, ch_keys, f"riemannian_occurrences", colors  # _{dts}",
)

# Print min-max channels
idx_max = np.argmax(abs(ch_norm))
idx_min = np.argmin(abs(ch_norm))
print("max: {:.2f}, ch: {}".format(ch_norm[idx_max], ch_keys[idx_max]))
print("min: {:.2f}, ch: {}".format(ch_norm[idx_min], ch_keys[idx_min]))
