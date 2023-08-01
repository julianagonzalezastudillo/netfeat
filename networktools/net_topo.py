import sys

# import networkx as nx
# from itertools import permutations
# from networkx.exception import NetworkXNoPath
import numpy as np
import re


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def channel_idx(ch_names, positions, print_ch=False):

    # Convert channel names to uppercase
    ch_names = [ch.upper() for ch in ch_names]

    RH_idx = []          # Right Hemisphere channel indices
    LH_idx = []          # Left Hemisphere channel indices
    CH_idx = []          # Central line channel indices
    CH_bis_idx = []      # Additional channel indices

    # Accumulate all left and central channels
    for i, ch in enumerate(ch_names):
        last_char = ch[-1]

        if last_char == 'H':
            if int(ch[-2]) % 2 != 0:
                LH_idx.append(i)
        elif last_char.isdigit() and int(last_char) % 2 != 0:
            LH_idx.append(i)
        elif last_char == 'Z':
            CH_idx.append(i)

    ch_miss = []
    # Find the homologous pair of channels and the closest central to the pair
    for i in LH_idx:
        ch_s = re.findall(r'\D+', ch_names[int(i)])  # string
        ch_n = re.findall(r'\d+', ch_names[int(i)])  # number

        if len(ch_s) == 2:
            rh = ch_s[0] + str(int(ch_n[0]) + 1) + ch_s[1]
        else:
            rh = ch_s[0] + str(int(ch_n[0]) + 1)
        ch = ch_s[0] + 'Z'

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
        print(bcolors.WARNING + "Warning: Channel {0} is not included in the list".format(ch_miss) + bcolors.ENDC)

    if print_ch:
        LH_names = [ch_names[index] for index in LH_idx]
        print('LH: {0}'.format(LH_names))
        RH_names = [ch_names[index] for index in RH_idx]
        print('RH: {0}'.format(RH_names))
        CH_names = [ch_names[index] for index in CH_bis_idx]
        print('CH: {0}'.format(CH_names))

    return np.array(RH_idx), np.array(LH_idx), np.array(CH_idx), np.array(CH_bis_idx)


def h_modules(X, RH_idx, LH_idx, CH_idx, CH_bis_idx):

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


def local_laterality(X, ch_names, positions, hemisphere=None):
    """
    order LEFT-RIGHT. Calculate local laterality for each node, considering both hemispheres
    X: connectivity matrix
    ch_names: channel names
    positions: channels positions in the same order as ch_names
    :return:
    """
    RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_names, positions)

    # Extract sub matrices for left hemisphere, right hemisphere, and central channels
    LH, _, _, RH, _, _, CH, _, _ = h_modules(X, RH_idx, LH_idx, CH_idx, CH_bis_idx)

    if hemisphere == 'left':
        lat_homol = np.sum(LH, axis=1) / np.sum(CH, axis=1)

    elif hemisphere == 'right':
        lat_homol = np.sum(RH, axis=1) / np.sum(CH, axis=1)

    else:
        # diference across paired of homologous nodes
        lat_homol = np.zeros(len(LH_idx) + len(RH_idx))
        lat_homol[:len(RH_idx)] = (np.sum(LH, axis=1) - np.sum(RH, axis=1)) / np.sum(CH, axis=1)
        lat_homol[len(RH_idx):] = (np.sum(RH, axis=1) - np.sum(LH, axis=1)) / np.sum(CH, axis=1)

    return lat_homol


def integration(X, ch_names, positions, hemisphere=None):
    """
    order LEFT-RIGHT. Calculate integration for each node, considering both hemispheres
    X: connectivity matrix
    ch_names: channel names
    positions: channels positions in the same order as ch_names
    :return:
    """
    RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_names, positions)
    LL, LC, LR, RR, RC, RL, CC, CL, CR = h_modules(X, RH_idx, LH_idx, CH_idx, CH_bis_idx)

    if hemisphere == 'left':
        # Compute integration for the left hemisphere
        num_L = LL.sum(axis=1) + LC.sum(axis=1) + LR.sum(axis=1)
        denom = CL.sum(axis=1) + CR.sum(axis=1) + CC.sum(axis=1)
        intg = num_L / denom

    elif hemisphere == 'right':
        # Compute integration for the right hemisphere
        num_R = RR.sum(axis=1) + RC.sum(axis=1) + RL.sum(axis=1)
        denom = CL.sum(axis=1) + CR.sum(axis=1) + CC.sum(axis=1)
        intg = num_R / denom

    else:
        # Compute integration for both hemispheres
        num_L = LL.sum(axis=1) + LC.sum(axis=1) + LR.sum(axis=1) - (RR.sum(axis=1) + RC.sum(axis=1) + RL.sum(axis=1))
        num_R = RR.sum(axis=1) + RC.sum(axis=1) + RL.sum(axis=1) - (LL.sum(axis=1) + LC.sum(axis=1) + LR.sum(axis=1))
        denom = CL.sum(axis=1) + CR.sum(axis=1) + CC.sum(axis=1)
        intg = np.concatenate([num_L / denom, num_R / denom])

    return intg


def segregation(X, ch_names, positions, hemisphere=None):
    """
    order LEFT-RIGHT. Calculate segregation for each node, considering both hemispheres
    X: connectivity matrix
    ch_names: channel names
    positions: channels positions in the same order as ch_names
    :return:
    """
    RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_names, positions)
    LL, LC, LR, RR, RC, RL, CC, CL, CR = h_modules(X, RH_idx, LH_idx, CH_idx, CH_bis_idx)

    if hemisphere == 'left':
        # Compute segregation for the left hemisphere
        num_L = np.sum(LL, axis=1) + np.sum(LC, axis=1) - np.sum(LR, axis=1)
        denom = np.sum(CL, axis=1) + np.sum(CR, axis=1) + np.sum(CC, axis=1)
        seg = num_L / denom

    elif hemisphere == 'right':
        # Compute segregation for the right hemisphere
        num_R = np.sum(RR, axis=1) + np.sum(RC, axis=1) - np.sum(RL, axis=1)
        denom = np.sum(CL, axis=1) + np.sum(CR, axis=1) + np.sum(CC, axis=1)
        seg = num_R / denom

    else:
        # Compute segregation for both hemispheres
        num_L = np.sum(LL, axis=1) + np.sum(LC, axis=1) - np.sum(LR, axis=1) - (np.sum(RR, axis=1) + np.sum(RC, axis=1) - np.sum(RL, axis=1))
        num_R = np.sum(RR, axis=1) + np.sum(RC, axis=1) - np.sum(RL, axis=1) - (np.sum(LL, axis=1) + np.sum(LC, axis=1) - np.sum(LR, axis=1))
        denom = np.sum(CL, axis=1) + np.sum(CR, axis=1) + np.sum(CC, axis=1)
        seg = np.concatenate([num_L / denom, num_R / denom])

    return seg
