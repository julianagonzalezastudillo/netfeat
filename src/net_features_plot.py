import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd

from networktools.net_topo import channel_idx, positions_matrix
from plottools.plot_positions import channel_pos
from plottools.plot_tools import colorbar, save_mat_file
from config import load_config, ConfigPath, LATERALIZATION_METRIC


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
colors = [
    [0.0, "#0045ff"],
    [0.125, "#4d7bfe"],
    [0.25, "#86aaf8"],
    [0.375, "#c0d5ee"],
    [0.5, "#ffffe0"],
    [0.625, "#ffcda5"],
    [0.75, "#ff966a"],
    [0.875, "#fa532d"],
    [1.0, "#d90000"],
]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
for name, metric in params["net_metrics"].items():
    print(metric)
    # Filter by selected metric and create dict of channels and t-values
    df_metric = df_t_test[df_t_test["metric"] == metric]

    # for each dataset or all datasets together
    for dts in np.append(df_metric.dataset.unique(), "all"):
        print(dts)
        if dts == "all":
            df_metric_dt = df_metric.loc[:]
            ch_names_dts = ch_keys
        else:
            df_metric_dt = df_metric.loc[df_metric["dataset"] == dts]
            ch_names_dts = params["ch_names"][dts]

        if metric in LATERALIZATION_METRIC:
            pos_dts = {key: value for key, value in pos.items() if key in ch_names_dts}
            pos_dts = np.array([value for value in pos_dts.values()]).reshape(-1, 2)
            RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_names_dts, pos_dts)
            ch_names = np.array([ch_names_dts[index] for index in LH_idx])
            fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
            xlim_max = 0
        else:
            ch_names = df_metric_dt.node.unique()
            fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
            xlim_max = max(positions[:, 0])
        ch_names = ch_names[~np.isin(ch_names, ["F9", "F10", "M1", "M2"])]

        # Assign position to each node
        ch_size = channel_size(df_metric_dt, ch_names, effect_size=False)
        ch_pos = np.array([pos[ch] for ch in ch_names])

        # Threshold for node names
        thresh = (
            2
            if metric in LATERALIZATION_METRIC
            else abs(ch_size[np.argsort(abs(ch_size))[-10:]][0])
        )

        # Define node size and max
        ch_size = channel_size(df_metric_dt, ch_names, effect_size=False)
        factor = 0.01 * fig.dpi * fig.get_size_inches()[0]

        # Plot t-values
        divnorm = matplotlib.colors.TwoSlopeNorm(
            vmin=-max(abs(ch_size)), vcenter=0.0, vmax=max(abs(ch_size))
        )
        thplot = ax.scatter(
            ch_pos[:, 0],
            ch_pos[:, 1],
            s=(abs(ch_size) * factor) ** 2,
            c=ch_size,
            marker=".",
            cmap=cmap,
            norm=divnorm,
            alpha=0.9,
            linewidths=0,
        )
        for i, ch in enumerate(ch_names):
            ax.text(
                ch_pos[i, 0],
                ch_pos[i, 1],
                ch,
                fontname="Arial",
                fontsize=6,
                horizontalalignment="center",
                verticalalignment="center",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_xlim(min(positions[:, 0]), xlim_max)
        ax.set_ylim(min(positions[:, 1]) * 1.1, max(positions[:, 1]) * 1.1)
        # plt.title(f"{dts} {metric}")
        colorbar(fig, thplot)
        plt.show()
        fig_name = ConfigPath.RES_DIR / f"stats/t_test_{metric}_{dts}.png"
        fig.savefig(fig_name, transparent=True)

        # Get 3D layout and save
        xyz = channel_pos(ch_names, dimension="3d", montage_type="standard") * 900
        norm = plt.Normalize(vmin=-max(abs(ch_size)), vmax=max(abs(ch_size)))
        rgb_values = cmap(norm(ch_size))

        # Select channel names to plot
        ch_names_ = np.array(ch_names, dtype=object)
        ch_names_[abs(ch_size) < thresh] = 0

        save_mat_file(
            ch_size,
            xyz,
            rgb_values,
            ch_names_,
            ConfigPath.RES_DIR / f"stats/t_test_{metric}_{dts}",
        )

        # Print min-max channels
        idx_max = np.argmax(abs(ch_size))
        idx_min = np.argmin(abs(ch_size))
        print("max: {:.2f}, ch: {}".format(ch_size[idx_max], ch_names[idx_max]))
        print("min: {:.2f}, ch: {}".format(ch_size[idx_min], ch_names[idx_min]))
