import sys

import networkx as nx
from itertools import permutations
from networkx.exception import NetworkXNoPath
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


def channel_idx(ch_names, positions):
    ch_names = [ch.upper() for ch in ch_names]
    RH_idx = []
    LH_idx = []
    LH_idx_aux = []
    CH_idx = []
    CH_bis_idx = []
    CH_idx_aux = []

    for i, ch in enumerate(ch_names):
        last_char = ch[-1]

        if last_char == 'H':
            if int(ch[-2]) % 2 != 0:
                LH_idx_aux.append(i)
        elif last_char.isdigit() and int(last_char) % 2 != 0:
            LH_idx_aux.append(i)
        elif last_char == 'Z':
            CH_idx_aux.append(i)

    ch_miss = []
    ch_disc = []
    for i in LH_idx_aux:
        ch_s = re.findall(r'\D+', ch_names[int(i)])  # string
        ch_n = re.findall(r'\d+', ch_names[int(i)])  # number

        if len(ch_s) == 2:
            rh = ch_s[0] + str(int(ch_n[0]) + 1) + ch_s[1]
        else:
            rh = ch_s[0] + str(int(ch_n[0]) + 1)
        ch = ch_s[0] + 'Z'

        if ch in ch_names:
            CH_idx = np.append(CH_idx, ch_names.index(ch))

        if ch in ch_names:
            CH_bis_idx = np.append(CH_bis_idx, ch_names.index(ch))
            RH_idx = np.append(RH_idx, ch_names.index(rh))
            LH_idx = np.append(LH_idx, ch_names.index(ch_names[int(i)]))
        else:
            # find closest
            dist = []
            for cz in CH_idx_aux:
                a = positions[int(cz)]
                b = positions[int(i)]
                dist = np.append(dist, np.linalg.norm(a - b))
            cz_idx = np.argmin(dist)
            CH_bis_idx = np.append(CH_bis_idx, CH_idx_aux[cz_idx])
            LH_idx = np.append(LH_idx, ch_names.index(ch_names[int(i)]))
            try:
                RH_idx = np.append(RH_idx, ch_names.index(rh))
            except ValueError:
                # print('Chanel {0} is not included in the montage'.format(rh))
                # print('Then channel {0} is not consider'.format(ch_names[int(i)]))
                LH_idx = LH_idx[:-1]
                CH_bis_idx = CH_bis_idx[:-1]
                pass

            # ch_miss = np.append(ch_miss, ch)
            # ch_disc = np.append(ch_disc, rh)
            # ch_disc = np.append(ch_disc, ch_names[int(i)])

    # if print_flag == True:
    #    print(bcolors.WARNING + "Warning: Channel {0} is not included in the list, then channels {1} are discarded for "
    #                            "the computation of laterality metrics".format(ch_miss, ch_disc) + bcolors.ENDC)

    return RH_idx.astype(int), LH_idx.astype(int), np.array(CH_idx_aux).astype(int), CH_bis_idx.astype(int)


def local_laterality(X, ch_names, positions, hemisphere=None):
    """
    order LEFT-RIGHT. Calculate local laterality for each node, considering both hemispheres
    X: connectivity matrix
    ch_names: channel names
    positions: channels positions in the same order as ch_names
    :return:
    """
    RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_names, positions)
    # LH_names = [ch_names[index] for index in LH_idx]
    # print('LH: {0}'.format(LH_names))
    # RH_names = [ch_names[index] for index in RH_idx]
    # print('RH: {0}'.format(RH_names))
    # CH_names = [ch_names[index] for index in CH_idx]
    # print('CH: {0}'.format(CH_names))

    # Extract sub matrices for left hemisphere, right hemisphere, and central channels
    LH = X[LH_idx, :][:, LH_idx]
    RH = X[RH_idx, :][:, RH_idx]
    CH = X[CH_bis_idx, :][:, CH_idx]

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
