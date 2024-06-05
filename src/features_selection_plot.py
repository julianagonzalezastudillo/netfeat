import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
from collections import Counter

from plottools.plot_positions import channel_pos
from plottools.plot_tools import save_mat_file
from plottools.plot_features import feature_plot_2d, normalize_channel_sizes
from config import (
    load_config,
    ConfigPath,
    DATASETS,
    EXCLUDE_CHANNELS,
    LATERALIZATION_METRIC,
)


def load_features(pipe, dataset_name):
    """Add a colorbar to a plot.

    Parameters
    ----------
    pipe : str
        Pipeline to load.

    dataset_name : str
        Dataset name.

    Returns
    -------
    ch: dict
        A dictionary containing the count of selected channels.
    """
    # Load all selected channels
    select_features = pd.read_csv(
        ConfigPath.RES_DIR
        / f"select_features/select_features_{dataset_name}_{pipeline}.csv"
    )
    if pipe == "RG+SVM":
        select_channels = [
            c
            for ch_cv in select_features["ch"]
            for c in ch_cv.strip("']['").split("' '")
        ]
    else:
        select_channels = [
            c
            for ch_cv in select_features["ch"]
            for c in ch_cv.strip("']['").split("', '")
        ]
    return select_channels


# Load params
params, paradigm = load_config()

# create dict with all the possible channels in all datasets
dt_ch_names = [params["ch_names"][dt.code] for dt in DATASETS]
ch_keys_all = np.unique(sum(dt_ch_names, []))

pipelines = ["RG+SVM"]
for name, metric in params["net_metrics"].items():
    pipelines.append(f"{params['fc']}+{name}+SVM")

for pipeline in pipelines:
    palette = params["features_palette"][pipeline]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", palette)
    ch_count_all = {ch_i: [] for ch_i in ch_keys_all}
    # ch_count_all = dict.fromkeys(ch_keys_all, 0)

    for dt in np.append(DATASETS, "all"):
        if dt == "all":
            dt_code = "all"
            filtered_ch_count_all = {
                key: value
                for key, value in ch_count_all.items()
                if key not in EXCLUDE_CHANNELS
            }
            ch_names = np.array(list(filtered_ch_count_all.keys()), dtype=object)
            ch_norm = np.array(
                [np.mean(values) for values in filtered_ch_count_all.values()]
            )
            # ch_count = ch_count_all
            # ch_names = np.array(list(ch_count.keys()), dtype=object)
        else:
            # Initialize counters
            ch_keys = params["ch_names"][dt.code]
            ch_count = dict.fromkeys(ch_keys, 0)

            # load selected features file
            dt_code = dt.code

            # all selected channels
            ch = load_features(pipeline, dt_code)

            # Count channels per dataset
            ch_count.update(
                {ch_i: ch_count[ch_i] + Counter(ch)[ch_i] for ch_i in Counter(ch)}
            )
            ch_names = np.array(list(ch_count.keys()), dtype=object)

            # Count channels for all datasets
            # ch_count_all.update(
            #     {ch_i: ch_count_all[ch_i] + ch.count(ch_i) for ch_i in np.unique(ch)}
            # )

            # Exclude channels
            ch_names = ch_names[~np.isin(ch_names, EXCLUDE_CHANNELS)]

            # Normalize by times it could have been selected == occurrences
            ch_norm, ch_names = normalize_channel_sizes(
                ch_count, dt, select_channels=ch_names
            )

            # Accumulate channels for all datasets
            # normalize for each dataset
            ch_norm_ = ch_norm / max(ch_norm)
            [
                ch_count_all[ch_i].append(ch_norm[idx])
                for idx, ch_i in enumerate(ch_names)
            ]

        print(f"{pipeline}: {dt_code}")
        # get electrodes positions
        ch_pos = channel_pos(ch_names, dimension="2d")

        # Create 2D figure
        fig, ax = feature_plot_2d(ch_names, ch_pos, ch_norm, cmap=cmap)
        plt.show()
        fig_name = f"occurrences_{pipeline}_{dt_code}"
        fig.savefig(
            ConfigPath.RES_DIR / f"select_features/plot/{fig_name}.png",
            transparent=False,
        )

        # Get 3D layout and save
        # Select channel names to plot
        thresh = 0.6
        n_select = 5 if metric in LATERALIZATION_METRIC else 10
        if not np.any(np.abs(ch_norm) >= thresh):
            thresh = abs(ch_norm[np.argsort(abs(ch_norm))[-5:]][0])
        ch_name_idx = np.where(np.abs(ch_norm) >= thresh)[0]
        ch_name_idx_sort = ch_name_idx[np.argsort(ch_norm[ch_name_idx])]

        save_mat_file(
            ch_norm, palette, ch_names, fig_name, ch_name_idx=ch_name_idx, min_zero=True
        )

        # Print ten channels with the highest values
        print(f"10 highest channels: {ch_names[ch_name_idx_sort]}")
        print(f"values: {ch_norm[ch_name_idx_sort]}")
