"""
=================================
            NETFEAT
=================================
Features study: This script performs statistical analysis on network metrics across all datasets and subjects.
Identify significant differences in network properties between classes using a two-sample t-test.
"""
import gzip
import numpy as np
import pandas as pd
import pickle
from scipy import stats

from config import load_config, ConfigPath, LATERALIZATION_METRIC
from moabb_settings import create_info_file
import networktools.net_topo as net_topo
from moabb.datasets import (
    BNCI2014001,
    Cho2017,
    Lee2019_MI,
    MunichMI,
    PhysionetMI,
    Shin2017A,
    Schirrmeister2017,
    Weibo2014,
    Zhou2016,
)


params, paradigm = load_config()
datasets = [
    BNCI2014001(),
    Cho2017(),
    Lee2019_MI(),
    PhysionetMI(),
    Shin2017A(accept=True),
    Schirrmeister2017(),
    Weibo2014(),
    Zhou2016(),
]

# Initialize data containers
df_list = []
for name, metric in params["net_metrics"].items():
    ch_keys = np.unique(sum(params["ch_names"].values(), []))
    ch_values = np.zeros(len(ch_keys))
    ch_dict_t_val = dict(zip(ch_keys, ch_values))
    ch_dict_p_val = dict(zip(ch_keys, ch_values))

    for dataset in datasets:
        for subject in dataset.subject_list:
            # Initialize per-subject data containers
            df_sub_list = []

            print(f"Dataset: {type(dataset).__name__}, sub: {subject}")

            ch_dict_l_sub = dict(zip(ch_keys, ch_values))
            ch_dict_r_sub = dict(zip(ch_keys, ch_values))

            # Get subject info file
            info = create_info_file(dataset, subject, paradigm)

            # Get subject network metric data
            file_name = (
                ConfigPath.RES_NET
                / f"netmetric_{dataset.code}_{str(subject).zfill(3)}_{'-'.join(list(info['event_id']))}_{metric}.gz"
            )
            with gzip.open(file_name, "r") as f:
                Xnet = pickle.load(f)
            Xnet = Xnet[metric]

            # Divide data by classes
            l_idx = np.where(info["classes"] == "left_hand")
            r_idx = np.where(info["classes"] == "right_hand")
            metric_l_sub = Xnet[l_idx]
            metric_r_sub = Xnet[r_idx]

            # Select channel names according to the type of network metric
            if metric in LATERALIZATION_METRIC:
                ch_pos = net_topo.positions_matrix("standard_1005", info["ch_names"])
                RH_idx, LH_idx, CH_idx, CH_bis_idx = net_topo.channel_idx(
                    info["ch_names"], ch_pos
                )
                LH_names = [info["ch_names"][index] for index in LH_idx]
                RH_names = [info["ch_names"][index] for index in RH_idx]
                ch_names_dataset = np.concatenate((LH_names, RH_names))
            else:
                ch_names_dataset = info["ch_names"]

            for ch_i, idx in zip(ch_names_dataset, range(len(ch_names_dataset))):
                ch_dict_l_sub[ch_i] = metric_l_sub[:, idx]
                ch_dict_r_sub[ch_i] = metric_r_sub[:, idx]

            # Perform t-test across classes for each subject per channel
            n_t_val, n_p_val = stats.ttest_ind(metric_r_sub, metric_l_sub)

            for ch_i, t_val, p_val in zip(ch_names_dataset, n_t_val, n_p_val):
                ch_dict_t_val[ch_i] = np.append(ch_dict_t_val[ch_i], t_val)
                ch_dict_p_val[ch_i] = np.append(ch_dict_p_val[ch_i], p_val)

            # Fill DataFrame for this subject
            df_sub_list.append(
                pd.DataFrame(
                    {
                        "dataset": [dataset.code] * len(ch_names_dataset),
                        "subject": [subject] * len(ch_names_dataset),
                        "class_1": ["right_hand"] * len(ch_names_dataset),
                        "class_2": ["left_hand"] * len(ch_names_dataset),
                        "metric": [metric] * len(ch_names_dataset),
                        "node": ch_names_dataset,
                        "t_val": n_t_val,
                        "p_val": n_p_val,
                    }
                )
            )

            # Concatenate per-subject DataFrames into one for this dataset
            df_list.append(pd.concat(df_sub_list, ignore_index=True))

# Concatenate all DataFrames into one for all metrics and datasets
df_t_test = pd.concat(df_list, ignore_index=True)

# Save the results to a CSV file
df_t_test.to_csv(ConfigPath.RES_CLASSIFY_DIR / "stats/net_t_test.csv", index=False)
