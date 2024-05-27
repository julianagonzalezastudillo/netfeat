import scipy.io as sio
import numpy as np
from collections import Counter
from .plot_positions import channel_pos
from src.config import ConfigPath, load_config, DATASETS


def hex_to_rgb(hex_color):
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hexadecimal to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return r, g, b


def save_mat_file(ch_size, rgb, ch_name, file_name, colorbar, names_idx=None):
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

    colorbar : list  of shape (n_colors, 2)
        Each row has the color position in the colorbar and the color code in hex.

    names_idx : {array-like} of shape (n_channels)
        Vector with indexes of nodes names to be plotted.

    Returns
    -------
    save .mat
    """

    # Find 3D position for each channel
    if names_idx is None:
        names_idx = np.arange(len(ch_name))

    # Translate hex to rgb for colorbar
    colorbar_rgb = []
    for c in np.arange(len(colorbar)):
        r, g, b = hex_to_rgb(colorbar[c][1])
        colorbar_rgb.append([colorbar[c][0], r, g, b])

    # Set ticks for colorbar
    max_abs_ch_size = max(abs(ch_size))
    colorbar_ticks = [0, 0.25, 0.5, 0.75, 1]

    if min(ch_size) < 0:
        colorbar_ticks_labels = [
            -round(max_abs_ch_size, 2),
            -round(max_abs_ch_size * 0.5, 2),
            0,
            round(max_abs_ch_size * 0.5, 2),
            round(max_abs_ch_size, 2),
        ]
    else:
        colorbar_ticks_labels = [round(max_abs_ch_size * i, 2) for i in colorbar_ticks]

    # Save to .mat file
    xyz = channel_pos(ch_name, dimension="3d", montage_type="standard") * 900
    values = {
        "Xnet": ch_size,
        "xyz": xyz,
        "color": rgb,
        "names": ch_name[names_idx],
        "names_idx": names_idx,
        "colorbar_rgb": colorbar_rgb,
        "colorbar_ticks": colorbar_ticks,
        "colorbar_ticks_labels": colorbar_ticks_labels,
    }

    sio.savemat(ConfigPath.RES_3DPLOT / f"{file_name}.mat", values)


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
