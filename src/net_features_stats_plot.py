import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import scipy.stats as stats

from networktools.net_topo import channel_idx, positions_matrix
from plottools.plot_positions import channel_pos
from plottools.plot_tools import save_mat_file
from plottools.plot_features import feature_plot_2d
from config import (
    load_config,
    ConfigPath,
    LATERALIZATION_METRIC,
    EXCLUDE_CHANNELS,
    P_VAL,
)


def channel_size(df, channel_names, effect_size=False):
    """Calculate the channel size for meta-analysis statistics

    Parameters
    ----------
    df : DataFrame
        containing the metrics for each channel.

    effect_size : bool, optional
        If True, computes the effect size using weights defined based on number of subjects, by default False.

    channel_names : {array-like} of shape (n_channels)
        Vector with nodes to be selected.

    Returns
    -------
    ch_size : list
        Mean sizes for each channel.
    """
    if effect_size:
        # Multiply each t-val by effect size
        dt_nsubs = df["nsub"].unique()
        weights = np.sqrt(list(dt_nsubs))
        weights = weights / weights.sum()

        for dataset, w in zip(df.dataset.unique(), weights):
            df.loc[df.dataset == dataset, "t_val"] = (
                w * df.loc[df.dataset == dataset, "t_val"]
            )
        ch_t_val = df.groupby("node")["t_val"].apply(list).to_dict()
        ch_size_ = [
            np.mean(ch_t_val[ch]) for ch in channel_names
        ]  # this is the sum including size effect
    else:
        # Compute mean across t-values for each channel
        ch_t_val = df.groupby("node")["t_val"].apply(list).to_dict()
        ch_size_ = [np.mean(ch_t_val[ch]) for ch in channel_names]

    return np.array(ch_size_)


# Load parameters
params, paradigm = load_config()

# Get electrodes positions
ch_keys = np.unique(sum(params["ch_names"].values(), []))
positions = channel_pos(ch_keys, dimension="2d")
pos = {key: pos for key, pos in zip(ch_keys, positions)}

# Load stat data
df_t_test = pd.read_csv(ConfigPath.RES_DIR / "stats/net_t_test.csv")

# Add number of subjects per dataset
df_t_test["nsub"] = None
for dt in df_t_test.dataset.unique():
    df_dt = df_t_test[df_t_test.dataset == dt]
    nsubs = df_dt.subject.unique().shape[0]
    df_t_test.loc[df_t_test.dataset == dt, "nsub"] = nsubs

# Define colors
palette = params["colorbar"]["net"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", palette)
for name, metric in params["net_metrics"].items():
    # Filter by selected metric and create dict of channels and t-values
    df_metric = df_t_test[df_t_test["metric"] == metric]
    print("*" * 100)
    # for each dataset or all datasets together
    for dts in np.append(df_metric.dataset.unique(), "all"):
        print(f"{metric}: {dts}")
        if dts == "all":
            df_metric_dt = df_metric.loc[:]
            ch_names_dts = ch_keys
        else:
            df_metric_dt = df_metric.loc[df_metric["dataset"] == dts]
            ch_names_dts = params["ch_names"][dts]

        # if metric in LATERALIZATION_METRIC:
        #     pos_dts = {key: value for key, value in pos.items() if key in ch_names_dts}
        #     pos_dts = np.array([value for value in pos_dts.values()]).reshape(-1, 2)
        #     RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_names_dts, pos_dts)
        #     ch_names = np.array([ch_names_dts[index] for index in LH_idx])
        #     fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
        #     xlim_max = 0
        # else:
        #     ch_names = df_metric_dt.node.unique()
        #     fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        #     xlim_max = max(positions[:, 0])
        ch_names = df_metric_dt.node.unique()
        xlim_max = max(positions[:, 0])
        ch_names = ch_names[~np.isin(ch_names, EXCLUDE_CHANNELS)]

        # Assign position to each node
        ch_size = channel_size(df_metric_dt, ch_names, effect_size=False)
        ch_pos = np.array([pos[ch] for ch in ch_names])

        # Threshold for node names (p-val < 0.05)
        n_subjects = 0
        for dt_ in np.unique(df_metric_dt["dataset"]):
            df = df_metric_dt.loc[df_metric_dt["dataset"] == dt_]
            n_subjects += len(np.unique(df["subject"]))
        thresh = stats.t.ppf(1 - P_VAL / 2, n_subjects)
        print(f"t-val: {thresh}")
        if not np.any(np.abs(ch_size) >= thresh):
            thresh = abs(ch_size[np.argsort(abs(ch_size))[-10:]][0])

        # Define node size and max
        ch_size = channel_size(df_metric_dt, ch_names, effect_size=False)

        # Plot t-values
        divnorm = matplotlib.colors.TwoSlopeNorm(
            vmin=-max(abs(ch_size)), vcenter=0.0, vmax=max(abs(ch_size))
        )
        fig, ax = feature_plot_2d(ch_names, ch_pos, ch_size, cmap=cmap, divnorm=divnorm)
        plt.show()
        fig_name = ConfigPath.RES_DIR / f"stats/t_test_{metric}_{dts}.png"
        fig.savefig(fig_name, transparent=True)

        # Get 3D layout and save
        norm = plt.Normalize(vmin=-max(abs(ch_size)), vmax=max(abs(ch_size)))
        rgb_values = cmap(norm(ch_size))

        # Select channel names to plot
        ch_name_idx = np.where(np.abs(ch_size) >= thresh)[0]
        ch_name_idx_sort = ch_name_idx[np.argsort(ch_size[ch_name_idx])]

        save_mat_file(
            ch_size,
            palette,
            ch_names,
            f"t_test_{metric}_{dts}",
            ch_name_idx=ch_name_idx,
        )

        # Print significant channels
        print(f"significant t-val: {ch_names[ch_name_idx_sort]}")
