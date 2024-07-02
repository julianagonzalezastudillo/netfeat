"""
Segregation analysis of connection of each hemisphere with the middle line.
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gzip
from scipy import stats

import networktools.net_topo as net_topo
from plottools.plot_positions import channel_pos
from plottools.plot_tools import save_mat_file
from plottools.plot_features import feature_plot_2d

from moabb_settings import create_info_file
from config import load_config, ConfigPath, DATASETS, EXCLUDE_CHANNELS


def load_fc(dataset, subject, event_id, fc):
    """Load functional connectivity matrices."""
    file_name = (
        ConfigPath.RES_FC
        / f"functionalconnectivity_{dataset.code}_{str(subject).zfill(3)}_{'-'.join(list(event_id))}_coh.gz"
    )
    with gzip.open(file_name, "r") as f:
        X = pickle.load(f)
    return X[fc]


def load_net(dataset, subject, event_id):
    """Load network properties."""
    file_name = (
        ConfigPath.RES_NET
        / f"netmetric_{dataset.code}_{str(subject).zfill(3)}_{'-'.join(list(event_id))}_segregation.gz"
    )
    with gzip.open(file_name, "r") as f:
        X = pickle.load(f)
    return X["segregation"]


def structured_connectivity(X, rh_idx, lh_idx, ch_idx, ch_bis_idx):
    """Arrange connectivity matrices within and across hemispheres,
    considering three different modules: left hemisphere, right hemisphere and central line.
    """
    n_trials, n_LH, n_RH = len(X), len(lh_idx), len(rh_idx)

    # Initialize empty arrays for connectivity matrices
    ll_sub, ll_lc_sub, lc_lr_sub, lr_sub = [
        np.zeros((n_trials, n_LH)) for _ in range(4)
    ]
    rr_sub, rr_rc_sub, rc_rl_sub, rl_sub = [
        np.zeros((n_trials, n_RH)) for _ in range(4)
    ]

    # Loop through each trial
    for i in range(n_trials):
        # Compute connectivity modules for the current trial
        ll_i, lc_i, lr_i, rr_i, rc_i, rl_i, _, _, _ = net_topo.h_modules(
            X[i, :, :], rh_idx, lh_idx, ch_idx, ch_bis_idx
        )

        # Loop through each channel in each hemisphere
        for j in range(n_RH):
            ll_sub[i, j] = np.sum(ll_i[j])
            ll_lc_sub[i, j] = np.sum(ll_i[j]) + np.sum(lc_i[j])
            lc_lr_sub[i, j] = np.sum(lc_i[j]) + np.sum(lr_i[j])
            lr_sub[i, j] = np.sum(lr_i[j])

            rr_sub[i, j] = np.sum(rr_i[j])
            rr_rc_sub[i, j] = np.sum(rr_i[j]) + np.sum(rc_i[j])
            rc_rl_sub[i, j] = np.sum(rc_i[j]) + np.sum(rl_i[j])
            rl_sub[i, j] = np.sum(rl_i[j])

    # Return hemisphere structured connectivity matrices
    return ll_sub, ll_lc_sub, lc_lr_sub, lr_sub, rr_sub, rr_rc_sub, rc_rl_sub, rl_sub


def paired_t_test(X, ch_names):
    """Perform paired t-tests and update t-value dictionaries."""

    # Get channel indices
    ch_pos = net_topo.positions_matrix("standard_1005", ch_names)
    rh_idx, lh_idx, ch_idx, ch_bis_idx = net_topo.channel_idx(ch_names, ch_pos)

    # Organized connectivity within and across hemispheres
    (
        ll_sub,
        ll_lc_sub,
        lc_lr_sub,
        lr_sub,
        rr_sub,
        rr_rc_sub,
        rc_rl_sub,
        rl_sub,
    ) = structured_connectivity(X, rh_idx, lh_idx, ch_idx, ch_bis_idx)

    # Initialize dictionaries to store t-values
    ll_t_val, rr_t_val, ll_lc_t_val, rr_rc_t_val = {}, {}, {}, {}

    # Perform paired t-tests for each channel in each hemisphere
    for ch in range(len(rh_idx)):
        # Perform paired t-tests for various connectivity modules
        ll_ch_t_val, _ = stats.ttest_rel(ll_sub[:, ch], lc_lr_sub[:, ch])
        rr_ch_t_val, _ = stats.ttest_rel(rr_sub[:, ch], rc_rl_sub[:, ch])
        ll_lc_ch_t_val, _ = stats.ttest_rel(ll_lc_sub[:, ch], lr_sub[:, ch])
        rr_rc_ch_t_val, _ = stats.ttest_rel(rr_rc_sub[:, ch], rl_sub[:, ch])

        # Get channel names for left and right hemispheres
        ll_ch_name = ch_names[lh_idx[ch]]
        rr_ch_name = ch_names[rh_idx[ch]]

        # Update dictionaries with t-values for respective modules
        ll_t_val[ll_ch_name] = np.append(ll_t_val.get(ll_ch_name, []), ll_ch_t_val)
        rr_t_val[rr_ch_name] = np.append(rr_t_val.get(rr_ch_name, []), rr_ch_t_val)
        ll_lc_t_val[ll_ch_name] = np.append(
            ll_lc_t_val.get(ll_ch_name, []), ll_lc_ch_t_val
        )
        rr_rc_t_val[rr_ch_name] = np.append(
            rr_rc_t_val.get(rr_ch_name, []), rr_rc_ch_t_val
        )

    return ll_t_val, rr_t_val, ll_lc_t_val, rr_rc_t_val


# Load parameters
params, paradigm = load_config()
palette = params["colorbar"]["net"]

# Initialize t-value dictionaries
ch_keys = np.unique(sum(params["ch_names"].values(), []))
ch_values = np.zeros(len(ch_keys))
LL_t_val = dict(zip(ch_keys, ch_values))
RR_t_val = dict(zip(ch_keys, ch_values))
LL_LC_t_val = dict(zip(ch_keys, ch_values))
RR_RC_t_val = dict(zip(ch_keys, ch_values))

for dt in DATASETS:
    print(f"Processing dataset: {dt.code}")
    for sub in dt.subject_list:
        # Load data
        info = create_info_file(dt, sub, paradigm)
        Xfc = load_fc(dt, sub, info["event_id"], params["fc"])

        # paired t-test across trials (for each sub)
        LL_t, RR_t, LL_LC_t, RR_RC_t = paired_t_test(Xfc, info["ch_names"])

        # Update t-value dictionaries
        for key, value in LL_t.items():
            LL_t_val[key] = np.append(LL_t_val[key], value)
        for key, value in RR_t.items():
            RR_t_val[key] = np.append(RR_t_val[key], value)
        for key, value in LL_LC_t.items():
            LL_LC_t_val[key] = np.append(LL_LC_t_val[key], value)
        for key, value in RR_RC_t.items():
            RR_RC_t_val[key] = np.append(RR_RC_t_val[key], value)


# Plot segregation influence of the middle line
for conditions in [
    ["LL_t_val", "RR_t_val"],
    ["LL_LC_t_val", "RR_RC_t_val"],
]:
    t_val = dict(zip(ch_keys, ch_values))
    for condition in conditions:
        t_val_aux = eval(condition)
        t_val.update(
            {ch: t_val_aux[ch] for ch in ch_keys if np.size(t_val_aux[ch]) > 1}
        )

    # Exclude channels
    ch_keys = ch_keys[~np.isin(ch_keys, EXCLUDE_CHANNELS)]

    # Set electrodes sizes and positions
    ch_size = np.array(
        [
            t_val[ch][1:].mean() if np.size(t_val[ch]) > 1 else t_val[ch]
            for ch in ch_keys
        ]
    )

    # Create 2D figure
    fig, ax = feature_plot_2d(ch_size, ch_keys, palette=palette, min_val="negative")
    plt.show()
    fig_name = ConfigPath.RES_DIR / f"stats/{condition}.png"
    fig.savefig(fig_name, transparent=True)

    save_mat_file(
        ch_size, palette, ch_keys, f"{condition}", ch_name_idx=[], min_val="negative"
    )

# Plot segregation
# for dt in DATASETS:
#     ch_names = params["ch_names"][dt.code]
#     ch_pos = channel_pos(ch_names, dimension="2d")
#     rh_idx, lh_idx, ch_idx, ch_bis_idx = net_topo.channel_idx(ch_names, ch_pos)
#     ch_count_dt = {ch: [] for ch in ch_names}
#     ch_dt_all = {ch: [] for ch in ch_names}
#     for sub in dt.subject_list:
#         # Load data
#         info = create_info_file(dt, sub, paradigm)
#         # y = info["classes"]
#         Xnet = load_net(dt, sub, info["event_id"])
#
#         for i, ch_ in enumerate(np.concatenate([lh_idx, rh_idx])):
#             ch = ch_names[ch_]
#             sub_mean = np.mean(Xnet[:, i])
#             ch_count_dt[ch].append(sub_mean)
#
#     for ch, value in ch_count_dt.items():
#         ch_dt_all[ch] = np.mean(value)
#
#     ch_keys = ch_dt_all.keys()
#     ch_size = np.array(list(ch_dt_all.values()))
#     fig, ax = feature_plot_2d(ch_size, ch_keys, palette=palette)
#     plt.show()
