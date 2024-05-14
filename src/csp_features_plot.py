import pickle
import gzip

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from moabb.paradigms import LeftRightImagery
from plottools.plot_positions import channel_pos
from plottools.plot_tools import colorbar

from config import load_config, ConfigPath, DATASETS


# Load configuration parameters
params = load_config()
csp_component_order = params["csp_component_order"]
ch_names = params["ch_names"]

paradigm = LeftRightImagery(fmin=params["fmin"], fmax=params["fmax"])  # for rh vs lh
mi_class = {"rh": 0, "lh": 1}
# min-var component --> csp['patterns'][:, 0]
# max-var component --> csp['patterns'][:, 1]

# create dict with all the possible channels in all datasets
ch_keys = np.unique(sum(ch_names.values(), []))
ch_dict = dict.fromkeys(ch_keys, [])
ch_dict_sub = dict.fromkeys(ch_keys, [])

# get electrodes positions
ch_pos = channel_pos(ch_keys)

# plot params
colors = [[0.0, "#fffcf0"], [0.7, "#ffe17a"], [1.0, "#ffc600"]]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

for csp_f_p in ["filters", "patterns"]:
    for class_name in ["rh", "lh"]:
        for dt in DATASETS:
            # Loop through datasets
            for subject in dt.subject_list:
                data = dt.get_data([subject])
                sessions = [list(sessions.keys()) for sub, sessions in data.items()][0]

                for session in sessions:
                    file_name = (
                        ConfigPath.RES_CSP
                        / f"csp_{dt.code}_sub{str(subject).zfill(3)}_{session}_{csp_component_order}.gz"
                    )
                    with gzip.open(file_name, "r") as f:
                        csp = pickle.load(f)
                        ch_names_dt = csp["info"].ch_names

                # select first or second component according to which class
                norm_csp = csp[csp_f_p][:, mi_class[class_name]] / max(
                    abs(csp[csp_f_p][:, mi_class[class_name]])
                )
                for ch_i, idx in zip(ch_names_dt, np.arange(len(ch_names_dt))):
                    ch_dict_sub[ch_i] = np.append(ch_dict_sub[ch_i], norm_csp[idx])
                    # print('{0}: {1}'.format(ch_i, ch_dict_sub[ch_i]))

        # compute absolute mean value for each channel across datasets
        for ch_i in ch_keys:
            ch_dict[ch_i] = np.abs(ch_dict_sub[ch_i]).mean()

        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

        ch_dict = {
            k: v if not np.isnan(v) else 0 for k, v in ch_dict.items()
        }  # replace nan by 0
        ch_size = np.fromiter(ch_dict.values(), dtype=float)  # to array
        ch_name = ch_dict.keys()

        # plot csp values
        # fixed max size
        ch_size_norm_to_plot = ch_size / max(abs(ch_size))
        im = plt.scatter(
            ch_pos[:, 0],
            ch_pos[:, 1],
            s=(abs(ch_size_norm_to_plot) * fig.dpi / 6) ** 2,
            c=ch_size,
            marker=".",
            cmap=cmap,
            alpha=0.9,
            linewidths=0,
            edgecolors="k",
        )

        # add channel names
        for i, ch in zip(np.arange(len(ch_name)), ch_name):
            plt.text(
                ch_pos[i, 0],
                ch_pos[i, 1],
                ch,
                fontsize=7,
                horizontalalignment="center",
                verticalalignment="center",
                c="k",
                fontname="Arial",
            )

        # plot handle
        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("off")
        plt.title("{0} {1}".format(csp_f_p, class_name))

        # Colorbar
        colorbar(fig, im)

        plt.show()
        fig_name = (
            ConfigPath.RES_CSP
            / f"plot/csp_{csp_f_p}_mean_across_sub_alpha_beta_{class_name}.png"
        )
        # fig.savefig(fig_name, transparent=True)

        # print 10 highest values channels
        print(f"{csp_f_p} {class_name}")
        print(sorted(zip(list(ch_dict.values()), list(ch_name)), reverse=True)[:10])
