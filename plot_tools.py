import numpy as np
from mne.viz.topomap import _get_pos_outlines
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


def channel_pos_2D(ch_names):
    montage = mne.channels.make_standard_montage('standard_1005')
    info = mne.create_info(montage.ch_names, sfreq=256, ch_types="eeg")
    info['dig'] = montage.dig
    picks = list(np.arange(0, len(montage.ch_names)))
    sphere = np.array([0, 0, 0, 0.95])
    pos, outlines = _get_pos_outlines(info, picks, sphere, to_sphere=True)
    ch_pos = []
    for ch in ch_names:
        idx = montage.ch_names.index(ch)
        ch_pos.append(pos[idx])

    ch_pos = np.array(ch_pos)
    return ch_pos
