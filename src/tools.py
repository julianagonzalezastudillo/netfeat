from contextlib import contextmanager
import sys, os
import scipy.io as sio
import numpy as np


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def save_mat_file(ch_size, xyz, rgb, ch_name, file_name):
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

    Returns
    -------
    save .mat
    """

    values = {
        "Xnet": ch_size,
        "xyz": xyz,
        "color": rgb,
        "names": ch_name[np.nonzero(ch_name)],
        "names_idx": np.nonzero(ch_name),
    }

    sio.savemat(f"{file_name}.mat", values)
