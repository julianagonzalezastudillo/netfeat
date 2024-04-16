import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from mne.viz.topomap import _get_pos_outlines
import mne
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import pathlib
from networktools.net_topo import channel_idx, positions_matrix
from plottools.plot_positions import channel_pos_2D


def colorbar(ax, scatter):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.02)
    cax_new = fig.add_axes([cax.get_position().x1 + 0.01,
                            cax.get_position().y0 + 0.035,
                            0.02,
                            cax.get_position().height - 0.07])
    cbar = plt.colorbar(scatter, cax=cax_new, shrink=0.7)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, size=0)
    plt.delaxes(cax)

lat_metrics = ['local_laterality', 'segregation', 'integration']
network_metrics = {'lat': 'local_laterality',
                   'seg': 'segregation',
                   'intg': 'integration',
                   'TS': 'strength'}

path = pathlib.Path().resolve()
jsonfile = "params.json"
if jsonfile in os.listdir(path):
    jsonfullpath = os.path.join(path, jsonfile)
    with open(jsonfullpath) as jsonfile:
        parameterDict = json.load(jsonfile)
ch_names = parameterDict['ch_names']
fmin = parameterDict['fmin']
fmax = parameterDict['fmax']

# Get electrodes positions
ch_keys = np.unique(sum(ch_names.values(), []))
positions = channel_pos_2D(ch_keys)
pos = {}
for i, key in enumerate(ch_keys):
    pos[key] = positions[i]

# Load stat data
df_t_test = pd.read_csv("results/stats/net_t_test.csv")

# Define colors
colors = [[0.0, '#4253D6'], [0.5, '#e1e8ed'], [1.0, '#EC6D5C']]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
for name, metric in network_metrics.items():
    print(metric)
    # Filter by selected metric and create dict of channels and t-values
    filtered_df = df_t_test[df_t_test['metric'] == metric]
    ch_t_val = filtered_df.groupby('node')['t_val'].apply(list).to_dict()

    if metric in lat_metrics:
        RH_idx, LH_idx, CH_idx, CH_bis_idx = channel_idx(ch_keys, positions)
        LH_names = [ch_keys[index] for index in LH_idx]
        ch_names_lh = LH_names
        fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
        factor = 12
    else:
        ch_names_lh = ch_t_val.keys()
        ch_names_lh = np.unique(sum(ch_names.values(), []))
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        factor = 20

    # Compute mean across t-values for each channel
    ch_size = []
    ch_pos = []
    for ch in ch_names_lh:
        ch_size.append(np.mean(ch_t_val[ch]))
        ch_pos.append(pos[ch])
    ch_pos = np.array(ch_pos)
    ch_size = np.array(ch_size)

    # Plot t-values
    divnorm = matplotlib.colors.TwoSlopeNorm(vmin=min(ch_size), vcenter=0., vmax=max(ch_size))
    thplot = plt.scatter(ch_pos[:, 0], ch_pos[:, 1], s=(abs(ch_size)*factor)**2, c=ch_size, marker=".", cmap=cmap,
                         norm=divnorm, alpha=0.9, linewidths=0)
    for i, ch in enumerate(ch_names_lh):
        plt.text(ch_pos[i, 0], ch_pos[i, 1], ch, fontname='Arial', fontsize=6,
                 horizontalalignment='center', verticalalignment='center')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    # plt.title('{0}'.format(metric))
    colorbar(ax, thplot)
    plt.show()
    fig_name = 'results/stats/t_test_{0}.png'.format(metric)
    fig.savefig(fig_name, transparent=True)

    # Print min-max channels
    idx_max = np.argmax(abs(ch_size))
    idx_min = np.argmin(abs(ch_size))
    print('max: {:.2f}, ch: {}'.format(ch_size[idx_max], ch_names_lh[idx_max]))
    print('min: {:.2f}, ch: {}'.format(ch_size[idx_min], ch_names_lh[idx_min]))
    # TODO: get max for dataset
    # TODO: treshold for lateralization

