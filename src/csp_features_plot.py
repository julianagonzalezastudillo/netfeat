"""
=================================
            NETFEAT
=================================

"""
import pickle
import gzip
import matplotlib.pyplot as plt
import numpy as np

import warnings
import moabb

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

# plot params
palette = params["colorbar"]["csp"]

for csp_f_p in ["filters", "patterns"]:
    for class_name in ["rh", "lh"]:
        # create dict with all the possible channels in all datasets
        ch_names = np.unique(sum(params["ch_names"].values(), []))
        ch_dict_all = dict.fromkeys(ch_names, [])
        for dt in np.append(DATASETS, "all"):
            # Loop through datasets
            if dt == "all":
                ch_dict = ch_dict_all
                dt_name = dt
            else:
                ch_dict_sub = dict.fromkeys(params["ch_names"][dt.code], [])
                for subject in dt.subject_list:
                    sessions = params["sessions"][dt.code]
                    dt_name = dt.code

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
                        ch_dict_all[ch_i] = np.append(ch_dict_all[ch_i], norm_csp[idx])
                    ch_dict = ch_dict_sub

            # %% Exclude channels
            ch_dict = {
                key: value
                for key, value in ch_dict.items()
                if key not in EXCLUDE_CHANNELS
            }

            # Compute absolute mean value for each channel across datasets
            for ch_i in ch_dict.keys():
                ch_dict[ch_i] = np.abs(ch_dict[ch_i]).mean()

            # Replace nan by 0
            ch_dict = {k: v if not np.isnan(v) else 0 for k, v in ch_dict.items()}
            ch_size = np.fromiter(ch_dict.values(), dtype=float)  # to array
            ch_name = list(ch_dict.keys())

            # Plot 2D and save
            title = f"{csp_f_p} {class_name} "
            fig, ax = feature_plot_2d(ch_size, ch_name, palette=palette, title=title)
            plt.show()
            fig_name = f"csp_{csp_f_p}_mean_{class_name}_{dt_name}"
            # fig.savefig(ConfigPath.RES_CSP / f"plot/{fig_name}.png", transparent=True)

            # Get 3D layout and save
            ch_name_idx = np.argsort(-ch_size)[:10]
            save_mat_file(
                ch_size,
                palette,
                np.array(ch_name, dtype=object),
                fig_name,
                ch_name_idx=ch_name_idx,
                min_zero=True,
            )

            # print 10 highest values channels
            print(f"{csp_f_p} {class_name}")
            print(sorted(zip(list(ch_dict.values()), list(ch_name)), reverse=True)[:10])
