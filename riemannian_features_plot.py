import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from collections import Counter
import os
from pathlib import Path
import pandas as pd
from moabb.datasets import (utils,
                            BNCI2014001,
                            Lee2019_MI,
                            Zhou2016,
                            MunichMI,
                            Shin2017A,
                            Cho2017,
                            Schirrmeister2017,
                            Weibo2014)
from plot_tools import channel_pos_2D
import json
import pathlib


path = pathlib.Path().resolve()
jsonfile = "params.json"
if jsonfile in os.listdir(path):
    jsonfullpath = os.path.join(path, jsonfile)
    with open(jsonfullpath) as jsonfile:
        parameterDict = json.load(jsonfile)
fmin = parameterDict['fmin']
fmax = parameterDict['fmax']
component_order = parameterDict['csp_component_order']
ch_names = parameterDict['ch_names']
cv = parameterDict['cv']

datasets = [BNCI2014001(), Cho2017(), Lee2019_MI(), Schirrmeister2017(), Weibo2014(), Zhou2016()]
# datasets = [BNCI2014001()]
pipeline = 'RG+SVM'

# create dict with all the possible channels in all datasets
ch_keys = np.unique(sum(ch_names.values(), []))
ch_dict = dict.fromkeys(ch_keys, 0)

# in order to normalized, we need to know
# how many times each channel could be selected: ch*dt_sub*dt_sessions*cv
ch_list = sum([ch_names[type(dt_).__name__] * len(dt_.subject_list) * dt_.n_sessions * 5 for dt_ in datasets], [])
ch_counts = Counter(ch_list)

for dt in datasets:
    # load selected features file
    filename = "results/select_features/select_features_{0}_{1}".format(dt.code, pipeline)
    fs = pd.read_csv(os.path.join(os.getcwd(), filename + '.csv'))

    # all selected channels
    ch = [c for ch_cv in fs['ch'] for c in ch_cv.strip("']['").split("' '")]

    # count channels
    ch_dict.update({ch_i: ch_dict[ch_i] + ch.count(ch_i) for ch_i in np.unique(ch)})
    print(dt.code)

fig = plt.figure(figsize = (7, 5), dpi=300)

# get electrodes positions
ch_pos = channel_pos_2D(ch_keys)

# normalize by times it could have been selected == occurrences
ch_norm = np.array([ch_dict[ch]/ch_counts[ch] if ch_counts[ch] != 0 else 0 for ch in ch_keys])
ch_size_norm_to_plot = ch_norm/max(abs(ch_norm))  # for normalize size

colors = [[0, '#e1e8ed'], [1.0, '#ff843c']]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
im = plt.scatter(ch_pos[:, 0], ch_pos[:, 1], s=(abs(ch_size_norm_to_plot) * fig.dpi/6) ** 2, c=ch_norm,
                 marker=".", cmap=cmap, alpha=1, linewidths=0, edgecolors='k')

for i, ch in zip(np.arange(len(ch_keys)), ch_keys):
    plt.text(ch_pos[i, 0], ch_pos[i, 1], ch, fontsize=7, horizontalalignment='center', verticalalignment='center',
             c='k', fontname='Arial')

plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.title('RG occurrences')

# colorbar
cb_ax = fig.add_axes([.92, .124, .03, .7])
cbar = fig.colorbar(im, orientation='vertical', cax=cb_ax)
cbar.outline.set_linewidth(0.2)
cbar.outline.set_edgecolor('grey')
cbar.ax.tick_params(labelsize=7, size=0)

plt.show()
fig_name = Path(os.path.join(os.getcwd(), 'results', 'riemannian_features', 'plot',
                             'riemannian_occurrences_mean_across_sub_alpha_beta.png'))

fig.savefig( fig_name, transparent = True )