import scipy.io as sio
import numpy as np
from .plot_positions import channel_pos
from src.config import ConfigPath


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


def save_mat_file(ch_size, rgb, ch_name, file_name, names_idx=None):
    """Save .mat to do 3D brain plots.

    Parameters
    ----------
    ch_size : {array-like} of shape (n_channels)
        Vector with nodes values.

    xyz : {array-like} of shape (n_channels)
        3D nodes positions.

    rgb : {array-like} of shape (n_channels, 4)
        4D matrix with nodes colors.

    ch_name : {array-like} of shape (n_channels)
        Vector with nodes names.

    file_name : string
        Name use to save .mat file.

    names_idx : {array-like} of shape (n_channels)
        Vector with indexes of nodes names to be plotted.

    Returns
    -------
    save .mat
    """

    # Find 3D position for each channel
    if names_idx is None:
        names_idx = np.arange(len(ch_name))

    xyz = channel_pos(ch_name, dimension="3d", montage_type="standard") * 900
    values = {
        "Xnet": ch_size,
        "xyz": xyz,
        "color": rgb,
        "names": ch_name[names_idx],
        "names_idx": names_idx,
    }

    sio.savemat(ConfigPath.RES_3DPLOT / f"{file_name}.mat", values)
