import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
from collections import Counter

from plottools.plot_positions import channel_pos
from plottools.plot_tools import save_mat_file
from plottools.plot_features import feature_plot_2d, normalize_channel_sizes
from config import load_config, ConfigPath, DATASETS, EXCLUDE_CHANNELS


# Load params
params, paradigm = load_config()
pipeline = "RG+SVM"

# create dict with all the possible channels in all datasets
dt_ch_names = [params["ch_names"][dt.code] for dt in DATASETS]
ch_keys_all = np.unique(sum(dt_ch_names, []))
ch_count_all = dict.fromkeys(ch_keys_all, 0)

for dt in np.append(DATASETS, "all"):
    if dt == "all":
        dt_code = "all"
        ch_count = ch_count_all
        ch_names = np.array(list(ch_count.keys()), dtype=object)
    else:
        # Initialize counters
        ch_keys = params["ch_names"][dt.code]
        ch_count = dict.fromkeys(ch_keys, 0)

        # load selected features file
        dt_code = dt.code
        fs = pd.read_csv(
            ConfigPath.RES_DIR
            / f"select_features/select_features_{dt_code}_{pipeline}.csv"
        )

        # all selected channels
        ch = [c for ch_cv in fs["ch"] for c in ch_cv.strip("']['").split("' '")]

        # Count channels per dataset
        ch_count.update(
            {ch_i: ch_count[ch_i] + Counter(ch)[ch_i] for ch_i in Counter(ch)}
        )
        ch_names = np.array(list(ch_count.keys()))

        # Count channels for all datasets
        ch_count_all.update(
            {ch_i: ch_count_all[ch_i] + ch.count(ch_i) for ch_i in np.unique(ch)}
        )

    # Exclude channels
    ch_names = ch_names[~np.isin(ch_names, EXCLUDE_CHANNELS)]

    # normalize by times it could have been selected == occurrences
    ch_norm, ch_names_ = normalize_channel_sizes(ch_count, dt, select_channels=ch_names)

    # get electrodes positions
    ch_pos = channel_pos(ch_names, dimension="2d")

    # Create 2D figure
    palette = params["colorbar"]["riemannian"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", palette)
    fig, ax = feature_plot_2d(ch_names_, ch_pos, ch_norm, cmap=cmap)
    plt.show()
    fig_name = (
        ConfigPath.RES_DIR
        / f"select_features/plot/riemannian_occurrences_{dt_code}.png"
    )
    fig.savefig(fig_name, transparent=True)

    # Get 3D layout and save
    ch_name_idx = np.argsort(-ch_norm)[:10]

    save_mat_file(
        ch_norm,
        palette,
        ch_names,
        f"riemannian_occurrences_{dt_code}",
        ch_name_idx=ch_name_idx,
    )

    # Print ten channels with the highest values
    print(f"10 highest channels: {ch_names[np.argsort(-ch_norm)[:20]]}")
