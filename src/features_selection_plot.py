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

# Create dict with all the possible channels in all datasets
dt_ch_names = [params["ch_names"][dt.code] for dt in DATASETS]
ch_keys_all = np.unique(sum(dt_ch_names, []))

# Prepare pipeline names
pipelines = ["PSD+SVM", "RG+SVM"]
+[f"{params['fc']}+{name}+SVM" for name, metric in params["net_metrics"].items()]

for pipeline in pipelines:
    palette = params["features_palette"][pipeline]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", palette)
    ch_count_all = {ch_i: [] for ch_i in ch_keys_all}

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
        else:
            dt_code = dt.code
            ch_names = np.array(list(params["ch_names"][dt.code]), dtype=object)
            ch_count = dict.fromkeys(ch_names, 0)

            # Load selected features from file
            ch = load_features(pipeline, dt_code)

            # Update channel counts
            ch_count.update(
                {ch_i: ch_count[ch_i] + Counter(ch)[ch_i] for ch_i in Counter(ch)}
            )

            # Exclude unwanted channels
            ch_names = ch_names[~np.isin(ch_names, EXCLUDE_CHANNELS)]

            # Normalize by times it could have been selected == occurrences
            ch_norm, ch_names = normalize_channel_sizes(
                ch_count, dt, select_channels=ch_names
            )

            # Accumulate channels for all datasets
            [
                ch_count_all[ch_i].append(ch_norm[idx])
                for idx, ch_i in enumerate(ch_names)
            ]

        print(f"{pipeline}: {dt_code}")

        # Get 3D layout and save
        # Select channel names to plot
        metric = pipeline.split("+")[-2]
        n_select = 10  # default selection
        if metric in params["net_metrics"]:
            if params["net_metrics"][metric] in LATERALIZATION_METRIC:
                n_select = 5
            metric = params["net_metrics"][metric]

        # if not np.any(np.abs(ch_norm) >= thresh):
        thresh = abs(ch_norm[np.argsort(abs(ch_norm))[-n_select:]][0])
        ch_name_idx = np.where(np.abs(ch_norm) >= thresh)[0]
        ch_name_idx_sort = ch_name_idx[np.argsort(ch_norm[ch_name_idx])]

        fig_name = f"occurrences_{metric}_{dt_code}"
        save_mat_file(
            ch_norm, palette, ch_names, fig_name, ch_name_idx=ch_name_idx, min_zero=True
        )

        # Create 2D figure
        # Get electrodes positions
        ch_pos = channel_pos(ch_names, dimension="2d")
        fig, ax = feature_plot_2d(ch_names, ch_pos, ch_norm, cmap=cmap)
        plt.show()
        fig.savefig(
            ConfigPath.RES_DIR / f"select_features/plot/{fig_name}.png",
            transparent=False,
        )

        # Print ten channels with the highest values
        print(f"10 highest channels: {ch_names[ch_name_idx_sort]}")
        print(f"values: {ch_norm[ch_name_idx_sort]}")
