import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from moabb.datasets import (utils,
                            BNCI2014001,
                            Lee2019_MI,
                            Zhou2016,
                            MunichMI,
                            Shin2017A,
                            Cho2017,
                            Schirrmeister2017,
                            Weibo2014)
import pickle
import gzip
from moabb.paradigms import LeftRightImagery
from plot_tools import channel_pos_2D
import matplotlib.colors
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

datasets = [BNCI2014001()]
datasets = [BNCI2014001(), Cho2017(), Lee2019_MI(), Schirrmeister2017(), Weibo2014(), Zhou2016()]

paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)  # for rh vs lh
mi_class = {'rh': 0,
            'lh': 1}
# min-var component --> csp['patterns'][:, 0]
# max-var component --> csp['patterns'][:, 1]

# create dict with all the possible channels in all datasets
ch_keys = np.unique(sum(ch_names.values(), []))
ch_dict = dict.fromkeys(ch_keys, [])
ch_dict_sub = dict.fromkeys(ch_keys, [])

# get electrodes positions
ch_pos = channel_pos_2D(ch_keys)

# plot params
colors = [[0.0, '#fffcf0'], [0.7, '#ffe17a'], [1.0, '#ffc600']]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

for csp_f_p in ['filters', 'patterns']:
    for class_name in ['rh', 'lh']:
        for dt in datasets:
            for subject in dt.subject_list:
                data = dt.get_data([subject])
                sessions = [list(sessions.keys()) for sub, sessions in data.items()][0]

                for session in sessions:
                    file_name = Path(os.path.join(path, 'results', 'csp_features', 'csp_{0}_sub{1}_{2}_{3}.gz'
                                                  .format(dt.code, str(subject).zfill(3), session, component_order)))
                    with gzip.open(file_name, "r") as f:
                        csp = pickle.load(f)
                        ch_names_dt = csp["info"].ch_names

                # select first or second component according to which class
                norm_csp = csp[csp_f_p][:, mi_class[class_name]]/max(abs(csp[csp_f_p][:, mi_class[class_name]]))
                for ch_i, idx in zip(ch_names_dt, np.arange(len(ch_names_dt))):
                    ch_dict_sub[ch_i] = np.append(ch_dict_sub[ch_i], norm_csp[idx])
                    # print('{0}: {1}'.format(ch_i, ch_dict_sub[ch_i]))

        # compute absolute mean value for each channel across datasets
        for ch_i in ch_keys:
            ch_dict[ch_i] = np.abs(ch_dict_sub[ch_i]).mean()

        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

        ch_dict = {k: v if not np.isnan(v) else 0 for k, v in ch_dict.items()}  # replace nan by 0
        ch_size = np.fromiter(ch_dict.values(), dtype=float)  # to array
        ch_name = ch_dict.keys()

        # plot csp values
        # fixed max size
        ch_size_norm_to_plot = ch_size/max(abs(ch_size))
        im = plt.scatter(ch_pos[:, 0], ch_pos[:, 1], s=(abs(ch_size_norm_to_plot)*fig.dpi/6)**2, c=ch_size, marker=".", cmap=cmap,
                         alpha=0.9, linewidths=0, edgecolors='k')

        # add channel names
        for i, ch in zip(np.arange(len(ch_name)), ch_name):
            plt.text(ch_pos[i, 0], ch_pos[i, 1], ch, fontsize=7, horizontalalignment='center',
                     verticalalignment='center', c='k', fontname='Arial')

        # plot handle
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.title('{0} {1}'.format(csp_f_p, class_name))

        # colorbar
        cb_ax = fig.add_axes([.92, .124, .03, .7])
        cbar = fig.colorbar(im, orientation='vertical', cax=cb_ax)
        cbar.outline.set_linewidth(0.2)
        cbar.outline.set_edgecolor('grey')
        cbar.ax.tick_params(labelsize=7, size=0)

        plt.show()
        fig_name = Path(os.path.join(os.getcwd(), 'results', 'csp_features', 'plot',
                                     'csp_{0}_mean_across_sub_alpha_beta_coh_{1}.png'.format(csp_f_p, class_name)))
        fig.savefig(fig_name, transparent=True)

        # print 10 highest values channels
        print('{0} {1}'.format(csp_f_p, class_name))
        print(sorted(zip(list(ch_dict.values()), list(ch_name)), reverse=True)[:10])




