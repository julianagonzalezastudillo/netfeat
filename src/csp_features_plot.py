import pickle
import gzip

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import math

import warnings
import moabb

from plottools.plot_positions import channel_pos
from plottools.plot_features import feature_plot_2d
from plottools.plot_tools import save_mat_file

from config import load_config, ConfigPath, DATASETS, EXCLUDE_CHANNELS


moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# Load configuration parameters
params, paradigm = load_config()

mi_class = {"rh": 0, "lh": 1}
# min-var component --> csp['patterns'][:, 0]
# max-var component --> csp['patterns'][:, 1]

# create dict with all the possible channels in all datasets
ch_names = np.unique(sum(params["ch_names"].values(), []))
ch_dict = dict.fromkeys(ch_names, [])

# plot params
palette = params["colorbar"]["csp"]
# palette = [
#     [0.0, "#ffffe0"],
#     [0.25, "#f4c95b"],
#     [0.5, "#e08f41"],
#     [0.75, "#ba586b"],
#     [1.0, "#673c86"],
# ]
# palette = [
#     [0.0, "#ffffe0"],
#     [0.25, "#F0E34C"],
#     [0.5, "#3D8B5F"],
#     [0.75, "#4070AA"],
#     [1.0, "#673C86"],
# ]
# palette = [
#     [0.0, "#FFFFFF"],
#     [0.25, "#a3ddf4"],
#     [0.5, "#63b1fd"],
#     [0.75, "#457dff"],
#     [1.0, "#0045ff"],
# ]
# palette = [
#     [0.0, "#ffffe0"],
#     [0.2, "#ddf4e7"],
#     [0.4, "#bae9ed"],
#     [0.6, "#74d1fa"],
#     [0.8, "#3e90ff"],
#     [1.0, "#0045ff"],
# ]
# palette = [
#     [0.0, "#ffffe0"],
#     [0.2, "#c3ee78"],
#     [0.4, "#8bd764"],
#     [0.6, "#49b2b0"],
#     [0.8, "#5378db"],
#     [1.0, "#5f37d4"],
# ]
palette = [
    [0.0, "#ffffe0"],
    [0.25, "#f8e8b0"],
    [0.5, "#e8c87d"],
    [0.75, "#cfa047"],
    [1.0, "#ad7100"],
]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", palette)

for csp_f_p in ["filters", "patterns"]:
    for class_name in ["rh", "lh"]:
        # Loop through datasets
        ch_dict_sub = dict.fromkeys(ch_names, [])
        for dt in DATASETS:
            for subject in dt.subject_list:
                sessions = params["sessions"][dt.code]

                # Loop through all sessions
                csp_f_p_sess = []
                for session in sessions:
                    file_name = (
                        ConfigPath.RES_CSP
                        / f"csp_{dt.code}_sub{str(subject).zfill(3)}_{session}_{params['csp_component_order']}.gz"
                    )

                    # Get the channel names from the CSP info
                    with gzip.open(file_name, "r") as f:
                        csp = pickle.load(f)
                    ch_names_dt = csp["info"].ch_names

                    # Accumulate sessions on first or second component according to which class
                    csp_f_p_sess.append(csp[csp_f_p][:, mi_class[class_name]])

                # Mean across sessions
                csp_f_p_sess_mean = np.mean(csp_f_p_sess, 0)

                # Normalize between 0 and 1
                norm_csp = csp_f_p_sess_mean / max(abs(csp_f_p_sess_mean))

                for idx, ch_i in enumerate(ch_names_dt):
                    ch_dict_sub[ch_i] = np.append(ch_dict_sub[ch_i], norm_csp[idx])
                    # print('{0}: {1}'.format(ch_i, ch_dict_sub[ch_i]))
        # %%
        # Exclude channels
        ch_dict = {
            key: value for key, value in ch_dict.items() if key not in EXCLUDE_CHANNELS
        }

        # Compute absolute mean value for each channel across datasets
        for ch_i in ch_dict.keys():
            ch_dict[ch_i] = np.abs(ch_dict_sub[ch_i]).mean()

        # Replace nan by 0
        ch_dict = {k: v if not np.isnan(v) else 0 for k, v in ch_dict.items()}
        ch_size = np.fromiter(ch_dict.values(), dtype=float)  # to array
        ch_name = list(ch_dict.keys())
        ch_pos = channel_pos(ch_name)

        # Plot 2D and save
        vmax, vmin = (0.5, 0.06) if csp_f_p == "patterns" else (0.45, 0.1)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        title = f"{csp_f_p} {class_name}"
        fig, ax = feature_plot_2d(
            ch_name, ch_pos, ch_size, cmap=cmap, divnorm=norm, title=title
        )
        plt.show()
        # fig_name = (
        #     ConfigPath.RES_CSP
        #     / f"plot/csp_{csp_f_p}_mean_across_sub_alpha_beta_{class_name}.png"
        # )
        # fig.savefig(fig_name, transparent=True)

        # Get 3D layout and save
        norm = plt.Normalize(vmin=min(ch_size), vmax=max(abs(ch_size)))
        rgb_values = cmap(norm(ch_size))
        ch_name_idx = np.argsort(-ch_size)[:10]

        save_mat_file(
            ch_size,
            palette,
            np.array(ch_name, dtype=object),
            f"csp_{csp_f_p}_mean_across_sub_alpha_beta_{class_name}",
            ch_name_idx=ch_name_idx,
        )

        # print 10 highest values channels
        print(f"{csp_f_p} {class_name}")
        print(sorted(zip(list(ch_dict.values()), list(ch_name)), reverse=True)[:10])
        print(f"{csp_f_p} {class_name} max: {math.ceil(max(ch_size) * 100) / 100}")
        print(f"{csp_f_p} {class_name} min: {math.floor(min(ch_size) * 100) / 100}")
