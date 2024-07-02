"""Estimation of homotopic pairs of nodes."""
import numpy as np
import re
import copy
import mne
from tools import bcolors


def positions_matrix(montage_name, ch_names):
    """
    Retrieve the 3D positions of specified channels from a given MNE montage.

    Parameters
    ----------
    montage_name : str
        The name of the MNE montage to use (e.g., "standard_1005").

    ch_names : list of str
        List of channel names for which to retrieve positions.

    Returns
    -------
    ch_pos : ndarray
        An array containing the 3D positions of the channels specified in `ch_names`.
    """

    # Create the standard montage using the provided montage name
    montage = mne.channels.make_standard_montage(montage_name)

    # Retrieve channel positions from the montage
    montage_ch_pos = list(montage._get_ch_pos().items())

    # Convert channel names to uppercase to ensure case-insensitive matching
    ch_names_upper = [ch.upper() for ch in ch_names]

    # Extract channel names and their positions from the montage
    montage_ch = [row[0].upper() for row in montage_ch_pos]
    montage_pos = np.array([row[1] for row in montage_ch_pos])

    # Find indices of the provided channel names in the montage
    montage_idx = [montage_ch.index(name) for name in ch_names_upper]

    # Retrieve the positions of the specified channels using their indices
    ch_pos = montage_pos[montage_idx]
    return ch_pos


def channel_idx(ch_names, positions, print_ch=False):
    """
    Identify indices of channels in different hemispheres and central positions.

    Parameters
    ----------
    ch_names : list of str
        List of channel names.

    positions : array-like
        Channel positions.

    print_ch : bool, optional
        Whether to print the missing channel names, by default False.

    Returns
    -------
    RH_idx : list of int
        Indices of right hemisphere channels.

    LH_idx : list of int
        Indices of left hemisphere channels.

    CH_idx : list of int
        Indices of central line channels.

    CH_bis_idx : list of int
        Indices of additional central channels corresponding to pairs of left-right channels.
    """

    # Convert channel names to uppercase
    ch_names = [ch.upper() for ch in ch_names]

    RH_idx = []  # Right Hemisphere channel indices
    LH_idx = []  # Left Hemisphere channel indices
    CH_idx = []  # Central line channel indices
    CH_bis_idx = []  # Additional channel indices

    # Accumulate all left and central channels
    for i, ch in enumerate(ch_names):
        last_char = ch[-1]

        if last_char == "H":
            if int(ch[-2]) % 2 != 0:
                LH_idx.append(i)
        elif last_char.isdigit() and int(last_char) % 2 != 0:
            LH_idx.append(i)
        elif last_char == "Z":
            CH_idx.append(i)

    ch_miss = []
    # Find the homologous pair of channels and the closest central to the pair
    LH_idx_ = copy.deepcopy(LH_idx)
    for i in LH_idx_:
        ch_s = re.findall(r"\D+", ch_names[int(i)])  # string
        ch_n = re.findall(r"\d+", ch_names[int(i)])  # number

        if len(ch_s) == 2:
            rh = ch_s[0] + str(int(ch_n[0]) + 1) + ch_s[1]
        else:
            rh = ch_s[0] + str(int(ch_n[0]) + 1)
        ch = ch_s[0] + "Z"

        if ch in ch_names:
            CH_bis_idx.append(ch_names.index(ch))
            RH_idx.append(ch_names.index(rh))
        else:
            # Find the closest channel index
            dist = [np.linalg.norm(positions[cz] - positions[i]) for cz in CH_idx]
            cz_idx = np.argmin(dist)
            CH_bis_idx.append(CH_idx[cz_idx])
            # If right hemisphere channel not found, discard current left hemisphere channel
            try:
                RH_idx.append(ch_names.index(rh))
            except ValueError:
                LH_idx.pop(LH_idx.index(i))
                CH_bis_idx.pop()
                ch_miss.append(ch_names[i])
                continue

    if ch_miss:
        print(
            bcolors.WARNING
            + "Warning: Channel {0} is not included in the list".format(ch_miss)
            + bcolors.ENDC
        )

    if print_ch:
        LH_names = [ch_names[index] for index in LH_idx]
        print("LH: {0}".format(LH_names))
        RH_names = [ch_names[index] for index in RH_idx]
        print("RH: {0}".format(RH_names))
        CH_names = [ch_names[index] for index in CH_bis_idx]
        print("CH: {0}".format(CH_names))

    return np.array(RH_idx), np.array(LH_idx), np.array(CH_idx), np.array(CH_bis_idx)


def channel_idx_MunichMI():
    lh_idx = [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        20,
        22,
        24,
        26,
        28,
        30,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
        48,
        50,
        52,
        54,
        56,
        58,
        60,
        66,
        68,
        70,
        72,
        74,
        76,
        78,
        80,
        82,
        84,
        86,
        88,
        90,
        92,
        94,
        98,
        100,
        102,
        104,
        106,
        108,
        110,
        112,
        114,
        116,
        118,
        120,
        122,
        124,
        126,
    ]
    rh_idx = [
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        21,
        23,
        25,
        27,
        29,
        31,
        35,
        37,
        39,
        41,
        43,
        45,
        47,
        49,
        51,
        53,
        55,
        57,
        59,
        61,
        67,
        69,
        71,
        73,
        75,
        77,
        79,
        81,
        83,
        85,
        87,
        89,
        91,
        93,
        97,
        99,
        101,
        103,
        105,
        107,
        109,
        111,
        113,
        115,
        117,
        119,
        121,
        123,
        125,
        127,
    ]
    ch_bis_idx = [
        95,
        17,
        31,
        31,
        18,
        32,
        17,
        31,
        18,
        62,
        63,
        62,
        63,
        62,
        63,
        31,
        31,
        61,
        61,
        62,
        63,
        64,
        17,
        31,
        18,
        61,
        62,
        63,
        64,
        61,
        61,
        61,
        17,
        62,
        31,
        63,
        18,
        18,
        18,
        64,
        64,
        64,
        32,
        32,
        17,
        17,
        17,
        17,
        62,
        62,
        62,
        31,
        31,
        31,
        31,
        18,
        18,
        18,
        18,
    ]
    ch_idx = [17, 18, 31, 32, 61, 62, 63, 64, 95, 96]
    return rh_idx, lh_idx, ch_idx, ch_bis_idx


def h_modules(X, LH_idx, RH_idx, CH_idx, CH_bis_idx):
    """Arrange connectivity modules for different hemispheric
    and central modules."""

    LL = X[LH_idx, :][:, LH_idx]
    LC = X[LH_idx, :][:, CH_idx]
    LR = X[LH_idx, :][:, RH_idx]

    RR = X[RH_idx, :][:, RH_idx]
    RC = X[RH_idx, :][:, CH_idx]
    RL = X[RH_idx, :][:, LH_idx]

    CC = X[CH_bis_idx, :][:, CH_idx]
    CR = X[CH_bis_idx, :][:, RH_idx]
    CL = X[CH_bis_idx, :][:, LH_idx]

    return LL, LC, LR, RR, RC, RL, CC, CL, CR


def local_laterality(X, LH_idx, RH_idx, CH_idx, CH_bis_idx, hemisphere=None):
    """
    Calculate the local laterality for each node, considering both hemispheres.

    Parameters
    ----------
    X : numpy.ndarray
        Connectivity matrix.

    LH_idx : list of int
        Indices of nodes in the left hemisphere.

    RH_idx : list of int
        Indices of nodes in the right hemisphere.

    CH_idx : list of int
        Indices of central nodes.

    CH_bis_idx : list of int
        Indices of additional central nodes.

    hemisphere : str, optional
        Hemisphere for which to compute laterality ('left', 'right', or None).
        If None, compute the laterality for both hemispheres by comparing homologous nodes.

    Returns
    -------
    numpy.ndarray
        Array of local laterality values for each node.
    """
    # RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_names, positions)

    # Extract sub matrices for left hemisphere, right hemisphere, and central channels
    LH, _, _, RH, _, _, CH, _, _ = h_modules(X, LH_idx, RH_idx, CH_idx, CH_bis_idx)

    if hemisphere == "left":
        lat_homol = np.sum(LH, axis=1) / np.sum(CH, axis=1)

    elif hemisphere == "right":
        lat_homol = np.sum(RH, axis=1) / np.sum(CH, axis=1)

    else:
        # Difference across paired of homologous nodes
        lat_homol = np.zeros(len(LH_idx) + len(RH_idx))
        lat_homol[: len(RH_idx)] = (np.sum(LH, axis=1) - np.sum(RH, axis=1)) / np.sum(
            CH, axis=1
        )
        lat_homol[len(RH_idx) :] = (np.sum(RH, axis=1) - np.sum(LH, axis=1)) / np.sum(
            CH, axis=1
        )

    return lat_homol


def integration(X, LH_idx, RH_idx, CH_idx, CH_bis_idx, hemisphere=None):
    """
    Calculate the integration for each node, considering both hemispheres.

    Parameters
    ----------
    X : numpy.ndarray
        Connectivity matrix.

    LH_idx : list of int
        Indices of nodes in the left hemisphere.

    RH_idx : list of int
        Indices of nodes in the right hemisphere.

    CH_idx : list of int
        Indices of central nodes.

    CH_bis_idx : list of int
        Indices of additional central nodes.

    hemisphere : str, optional
        Hemisphere for which to compute integration ('left', 'right', or None).
        If None, compute the integration for both hemispheres by comparing homologous nodes.

    Returns
    -------
    numpy.ndarray
        Array of integration values for each node.
    """
    # RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_names, positions)
    LL, LC, LR, RR, RC, RL, CC, CL, CR = h_modules(
        X, LH_idx, RH_idx, CH_idx, CH_bis_idx
    )

    if hemisphere == "left":
        # Compute integration for the left hemisphere
        num_L = LL.sum(axis=1) + LC.sum(axis=1) + LR.sum(axis=1)
        denom = CL.sum(axis=1) + CR.sum(axis=1) + CC.sum(axis=1)
        intg = num_L / denom

    elif hemisphere == "right":
        # Compute integration for the right hemisphere
        num_R = RR.sum(axis=1) + RC.sum(axis=1) + RL.sum(axis=1)
        denom = CL.sum(axis=1) + CR.sum(axis=1) + CC.sum(axis=1)
        intg = num_R / denom

    else:
        # Compute integration for both hemispheres
        num_L = (
            LL.sum(axis=1)
            + LC.sum(axis=1)
            + LR.sum(axis=1)
            - (RR.sum(axis=1) + RC.sum(axis=1) + RL.sum(axis=1))
        )
        num_R = (
            RR.sum(axis=1)
            + RC.sum(axis=1)
            + RL.sum(axis=1)
            - (LL.sum(axis=1) + LC.sum(axis=1) + LR.sum(axis=1))
        )
        denom = CL.sum(axis=1) + CR.sum(axis=1) + CC.sum(axis=1)
        intg = np.concatenate([num_L / denom, num_R / denom])

    return intg


def segregation(X, LH_idx, RH_idx, CH_idx, CH_bis_idx, hemisphere=None):
    """
    Calculate the segregation for each node, considering both hemispheres.

    Parameters
    ----------
    X : numpy.ndarray
        Connectivity matrix.

    LH_idx : list of int
        Indices of nodes in the left hemisphere.

    RH_idx : list of int
        Indices of nodes in the right hemisphere.

    CH_idx : list of int
        Indices of central nodes.

    CH_bis_idx : list of int
        Indices of additional central nodes.

    hemisphere : str, optional
        Hemisphere for which to compute segregation ('left', 'right', or None).
        If None, compute the segregation for both hemispheres by comparing homologous nodes.

    Returns
    -------
    numpy.ndarray
        Array of segregation values for each node.
    """
    # RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_names, positions)
    LL, LC, LR, RR, RC, RL, CC, CL, CR = h_modules(
        X, LH_idx, RH_idx, CH_idx, CH_bis_idx
    )

    if hemisphere == "left":
        # Compute segregation for the left hemisphere
        num_L = np.sum(LL, axis=1) + np.sum(LC, axis=1) - np.sum(LR, axis=1)
        denom = np.sum(CL, axis=1) + np.sum(CR, axis=1) + np.sum(CC, axis=1)
        seg = num_L / denom

    elif hemisphere == "right":
        # Compute segregation for the right hemisphere
        num_R = np.sum(RR, axis=1) + np.sum(RC, axis=1) - np.sum(RL, axis=1)
        denom = np.sum(CL, axis=1) + np.sum(CR, axis=1) + np.sum(CC, axis=1)
        seg = num_R / denom

    else:
        # Compute segregation for both hemispheres
        num_L = (
            np.sum(LL, axis=1)
            + np.sum(LC, axis=1)
            - np.sum(LR, axis=1)
            - (np.sum(RR, axis=1) + np.sum(RC, axis=1) - np.sum(RL, axis=1))
        )
        num_R = (
            np.sum(RR, axis=1)
            + np.sum(RC, axis=1)
            - np.sum(RL, axis=1)
            - (np.sum(LL, axis=1) + np.sum(LC, axis=1) - np.sum(LR, axis=1))
        )
        denom = np.sum(CL, axis=1) + np.sum(CR, axis=1) + np.sum(CC, axis=1)
        seg = np.concatenate([num_L / denom, num_R / denom])

    return seg
