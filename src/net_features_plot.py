import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from networktools.net_topo import channel_idx, positions_matrix
from plottools.plot_positions import channel_pos_2d
from config import load_config, ConfigPath, LATERALIZATION_METRIC


def colorbar(ax, scatter):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02)
    cax_new = fig.add_axes(
        [
            cax.get_position().x1 + 0.01,
            cax.get_position().y0 + 0.035,
            0.02,
            cax.get_position().height - 0.07,
        ]
    )
    cbar = plt.colorbar(scatter, cax=cax_new, shrink=0.7)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, size=0)
    plt.delaxes(cax)


def channel_size(df, effect_size=False):
    """Calculate the channel size for meta-analysis statistics

    Parameters
    ----------
    df : DataFrame
        containing the metrics for each channel.

    effect_size : bool, optional
        If True, computes the effect size using weights defined based on number of subjects, by default False.

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
            np.mean(ch_t_val[ch]) for ch in ch_names_lh
        ]  # this is the sum including size effect
    else:
        # Compute mean across t-values for each channel
        ch_t_val = df.groupby("node")["t_val"].apply(list).to_dict()
        ch_size_ = [np.mean(ch_t_val[ch]) for ch in ch_names_lh]

    return ch_size_


params, paradigm = load_config()

# Get electrodes positions
ch_keys = np.unique(sum(params["ch_names"].values(), []))
positions = channel_pos_2d(ch_keys)
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

    if metric in LATERALIZATION_METRIC:
        RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_keys, positions)
        LH_names = [ch_keys[index] for index in LH_idx]
        ch_names_lh = LH_names
        fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    else:
        ch_names_lh = np.unique(sum(params["ch_names"].values(), []))
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    ch_size = channel_size(df_metric, effect_size=False)
    cte_size = 0.01

    # Define node max size
    factor = cte_size * fig.dpi * fig.get_size_inches()[0]

    ch_pos = np.array([pos[ch] for ch in ch_names_lh])
    ch_size = np.array(ch_size)

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
    for i, ch in enumerate(ch_names_lh):
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
    # plt.title('{0}'.format(metric))
    colorbar(ax, thplot)
    plt.show()
    fig_name = "results/stats/t_test_{0}.png".format(metric)
    # fig.savefig(fig_name, transparent=True)

    # Print min-max channels
    idx_max = np.argmax(abs(ch_size))
    idx_min = np.argmin(abs(ch_size))
    print("max: {:.2f}, ch: {}".format(ch_size[idx_max], ch_names_lh[idx_max]))
    print("min: {:.2f}, ch: {}".format(ch_size[idx_min], ch_names_lh[idx_min]))
    # TODO: get max for dataset
    # TODO: treshold for lateralization
