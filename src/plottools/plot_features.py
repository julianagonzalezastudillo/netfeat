import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from collections import Counter

from .plot_positions import channel_pos
from src.config import load_config, DATASETS


def colorbar(fig, scatter):
    """Add a colorbar to a plot.

    Parameters
    ----------
        fig : matplotlib.fig
        The main axes object to which the colorbar will be attached.

        scatter : matplotlib.pyplot.scatter
        The scatter plot object for which the colorbar is being added.

    Returns
    -------
        matplotlib.colorbar.Colorbar:
        The colorbar object that was added to the plot.
    """
    cb_ax = fig.add_axes([0.91, 0.145, 0.02, 0.7])
    cbar = fig.colorbar(scatter, orientation="vertical", cax=cb_ax)
    cbar.outline.set_linewidth(0.2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, size=0)


def feature_plot_2d(ch_names, ch_pos, ch_size, cmap=None, divnorm=None):
    """Plot scores for all pipelines and all datasets

    Parameters
    ----------
    ch_names: list of str
        Names of the channels.
    ch_pos: array-like, shape (n_channels, 2)
        Positions of the channels in 2D.
    ch_size: array-like, shape (n_channels,)
        Sizes of the channels for the plot.
    cmap: Colormap, optional
        Colormap to use for the scatter plot. If None, a default colormap is used.
    divnorm: Normalize, optional
        Normalization for the color scale. If None, a default normalization is used.

    Returns
    -------
    fig: Figure
        Matplotlib figure handle.
    ax: Axes
        Matplotlib axes handle.
    """

    # Create 2D figure
    dpi = 300
    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    if cmap is None:
        colors = [[0, "#e1e8ed"], [1.0, "#EC6D5C"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    ch_size_ = (
        abs(ch_size / max(abs(ch_size))) * dpi / 6
    ) ** 2  # for normalize size between 0, 1
    thplot = plt.scatter(
        ch_pos[:, 0],
        ch_pos[:, 1],
        s=ch_size_,
        c=ch_size,
        marker=".",
        cmap=cmap,
        norm=divnorm,
        alpha=1,
        linewidths=0,
        edgecolors="k",
    )

    for i, ch in enumerate(ch_names):
        ax.text(
            ch_pos[i, 0],
            ch_pos[i, 1],
            ch,
            fontsize=6,
            horizontalalignment="center",
            verticalalignment="center",
            c="k",
            fontname="Arial",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Get the same maximum size for all plots
    params, paradigm = load_config()
    ch_keys = np.unique(sum(params["ch_names"].values(), []))
    positions = channel_pos(ch_keys, dimension="2d")
    ax.set_xlim(min(positions[:, 0]), max(positions[:, 0]))
    ax.set_ylim(min(positions[:, 1]) * 1.1, max(positions[:, 1]) * 1.1)
    # plt.title(f"{dts} {metric}")

    # Add colorbar
    colorbar(fig, thplot)
    return fig, ax


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
    params, paradigm = load_config()
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


def normalize_channel_sizes(count, dataset, select_channels=None):
    """Count how many times each channel could be selected: ch*dt_sub*dt_sessions*cv

    Parameters
    ----------
    count : dict
        Dictionary containing all the channel counts.

    dataset : dataset instance
        The dataset to count channels for.

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

    # Dictionary containing all the maximum possible channel counts
    ch_total = total_channels(dataset)

    # Compute normalized size
    ch_size_norm = np.array(
        [count[c] / ch_total[c] if ch_total[c] != 0 else 0 for c in channels]
    )
    return ch_size_norm
