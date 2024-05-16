import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
from collections import Counter

from plottools.plot_positions import channel_pos
from plottools.plot_tools import colorbar, save_mat_file
from config import load_config, ConfigPath, DATASETS, EXCLUDE_CHANNELS


def total_channels(dataset):
    """Count how many times each channel could be selected: ch*dt_sub*dt_sessions*cv

    Parameters
    ----------
    dataset : dataset instance
        The dataset to count channels for.

    Returns
    -------
    channel_counts : dict
        A dictionary containing the count of how many times each channel
        could be selected.
    """
    if dataset == "all":
        channel_list = [
            item
            for dt_ in DATASETS
            for item in params["ch_names"][dt_.code]
            * len(dt_.subject_list)
            * dt_.n_sessions
            * params["cv"]
        ]
    else:
        channel_list = (
            params["ch_names"][dataset.code]
            * len(dataset.subject_list)
            * dataset.n_sessions
            * params["cv"]
        )

    # Count occurrences of each channel in the channel list
    channel_total = Counter(channel_list)

    return dict(channel_total)


def normalize_channel_sizes(count, total, select_channels=None):
    """Count how many times each channel could be selected: ch*dt_sub*dt_sessions*cv

    Parameters
    ----------
    count : dict
        Dictionary containing all the channel counts.

    total : dict
        Dictionary containing all the maximum possible channel counts.

    select_channels : list
        Array containing a specific list of channels to be selected.

    Returns
    -------
    ch_norm : array-like (n_channels,)
        Array containing the normalized channel sizes.

    """
    if select_channels is None:
        channels = np.array(list(count.keys()))
    else:
        channels = select_channels
    ch_norm = np.array(
        [count[ch] / total[ch] if total[ch] != 0 else 0 for ch in channels]
    )
    return ch_norm


# Load params
params, paradigm = load_config()
pipeline = "RG+SVM"

# create dict with all the possible channels in all datasets
ch_keys_all = np.unique(sum(params["ch_names"].values(), []))
ch_count_all = dict.fromkeys(ch_keys_all, 0)

for dt in np.append(DATASETS, "all"):
    if dt == "all":
        dt_code = "all"
        ch_count = ch_count_all
        ch_names = np.array(list(ch_count.keys()))
    else:
        # load selected features file
        dt_code = dt.code
        fs = pd.read_csv(
            ConfigPath.RES_DIR
            / f"select_features/select_features_{dt_code}_{pipeline}.csv"
        )

        # all selected channels
        ch = [c for ch_cv in fs["ch"] for c in ch_cv.strip("']['").split("' '")]

        # Count channels per dataset
        ch_count = dict(Counter(ch))
        ch_names = np.array(list(ch_count.keys()))

        # Count channels for all datasets
        ch_count_all.update(
            {ch_i: ch_count_all[ch_i] + ch.count(ch_i) for ch_i in np.unique(ch)}
        )

    # Exclude channels
    ch_names = ch_names[~np.isin(ch_names, EXCLUDE_CHANNELS)]

    # normalize by times it could have been selected == occurrences
    ch_total = total_channels(dt)
    ch_norm = normalize_channel_sizes(ch_count, ch_total, select_channels=ch_names)
    ch_size = ch_norm / max(abs(ch_norm))  # for normalize size between 0, 1

    # get electrodes positions
    ch_pos = channel_pos(ch_names, dimension="2d")

    # Create 2D figure
    fig = plt.figure(figsize=(7, 5), dpi=300)

    # Define colors
    colors = params["colorbar"]["riemannian"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    im = plt.scatter(
        ch_pos[:, 0],
        ch_pos[:, 1],
        s=(abs(ch_size) * fig.dpi / 6) ** 2,
        c=ch_norm,
        marker=".",
        cmap=cmap,
        alpha=1,
        linewidths=0,
        edgecolors="k",
    )

    for i, ch in zip(np.arange(len(ch_names)), ch_names):
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
        / f"riemannian_features/plot/riemannian_occurrences_{dt_code}.png"
    )
    # fig.savefig(fig_name, transparent=True)

    # Get 3D layout and save
    norm = plt.Normalize(vmin=0, vmax=max(abs(ch_norm)))
    rgb_values = cmap(norm(ch_norm))
    save_mat_file(
        ch_norm,
        rgb_values,
        ch_names,
        f"riemannian_occurrences_{dt_code}",
        colors,  # _{dts}"
    )

    # Print min-max channels
    idx_max = np.argmax(abs(ch_norm))
    idx_min = np.argmin(abs(ch_norm))
    print("max: {:.2f}, ch: {}".format(ch_norm[idx_max], ch_names[idx_max]))
    print("min: {:.2f}, ch: {}".format(ch_norm[idx_min], ch_names[idx_min]))
