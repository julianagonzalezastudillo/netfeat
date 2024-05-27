import matplotlib.pyplot as plt
import matplotlib.colors


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


def feature_plot_2d(ch_names, ch_pos, ch_size, ch_color, cmap=None, divnorm=None):
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
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    if cmap is None:
        colors = [[0, "#e1e8ed"], [1.0, "#EC6D5C"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    thplot = plt.scatter(
        ch_pos[:, 0],
        ch_pos[:, 1],
        s=ch_size,
        c=ch_color,
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
    # ax.set_xlim(min(positions[:, 0]), max(positions[:, 0]))
    # ax.set_ylim(min(positions[:, 1]) * 1.1, max(positions[:, 1]) * 1.1)
    # plt.title(f"{dts} {metric}")
    colorbar(fig, thplot)
    return fig, ax
