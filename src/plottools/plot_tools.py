import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.io as sio
from .plot_positions import channel_pos
from src.config import ConfigPath


def hex_to_rgb(hex_color):
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hexadecimal to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return r, g, b


def save_mat_file(
    ch_size, palette, ch_name, file_name, ch_name_idx=None, min_zero=False
):
    """Save .mat to do 3D brain plots.

    Parameters
    ----------
    ch_size : {array-like} of shape (n_channels)
        Vector with nodes values.

    palette : list  of shape (n_colors, 2)
        Each row has the color position in the colorbar and the color code in hex.

    ch_name : {array-like} of shape (n_channels)
        Vector with nodes names.

    file_name : string
        Name use to save .mat file.

    ch_name_idx : {array-like} of shape (n_channels)
        Vector with indexes of nodes names to be plotted.

    Returns
    -------
    save .mat
    """

    # Translate hex to rgb for colorbar
    palette_rgb = []
    for c in np.arange(len(palette)):
        r, g, b = hex_to_rgb(palette[c][1])
        palette_rgb.append([palette[c][0], r, g, b])

    # Colorbar
    # Set ticks
    colorbar_ticks = [0, 0.25, 0.5, 0.75, 1]

    # Set ticks labels
    max_abs_ch_size = max(abs(ch_size))
    if min(ch_size) < 0:
        colorbar_ticks_labels = [
            -round(max_abs_ch_size, 2),
            -round(max_abs_ch_size * 0.5, 2),
            0,
            round(max_abs_ch_size * 0.5, 2),
            round(max_abs_ch_size, 2),
        ]
        norm = plt.Normalize(vmin=-max(abs(ch_size)), vmax=max(abs(ch_size)))
    else:
        if not min_zero:
            min_ch_size = min(ch_size)
            ticks = np.linspace(min_ch_size, max_abs_ch_size, 5)
            colorbar_ticks_labels = np.round(ticks, 2)
            norm = plt.Normalize(vmin=min(ch_size), vmax=max(abs(ch_size)))
        else:
            colorbar_ticks_labels = [
                round(max_abs_ch_size * i, 2) for i in colorbar_ticks
            ]
            norm = plt.Normalize(vmin=0, vmax=max(abs(ch_size)))

    # Create colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", palette)
    rgb = cmap(norm(ch_size))

    # Find 3D position for each channel
    xyz = channel_pos(ch_name, dimension="3d", montage_type="standard") * 900

    # Set channel names to all if not selection
    if ch_name_idx is None:
        ch_name_idx = np.arange(len(ch_name))

    # Save to .mat file
    values = {
        "Xnet": ch_size,
        "xyz": xyz,
        "color": rgb,
        "names": ch_name[ch_name_idx],
        "names_idx": ch_name_idx,
        "colorbar_rgb": palette_rgb,
        "colorbar_ticks": colorbar_ticks,
        "colorbar_ticks_labels": colorbar_ticks_labels,
    }

    sio.savemat(ConfigPath.RES_3DPLOT / f"{file_name}.mat", values)
