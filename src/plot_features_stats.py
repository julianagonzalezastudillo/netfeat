"""
============================================
 5.2. NETFEAT - Plot feature stats
============================================

Group-averaged t-values, contrasting RMI versus LMI in the α-β band for PSD and network properties.
Performed for each dataset and across datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import multipletests

from plottools.plot_positions import channel_pos
from plottools.plot_tools import save_mat_file
from plottools.plot_features import feature_plot_2d
from config import (
    load_config,
    ConfigPath,
    EXCLUDE_CHANNELS,
    P_VAL,
)


def channel_size(df, channel_names, effect_size=False):
    """Calculate the channel size for meta-analysis statistics

    Parameters
    ----------
    df : DataFrame
        containing the metrics for each channel.

    effect_size : bool, optional
        If True, computes the effect size using weights defined based on number of subjects, by default False.

    channel_names : {array-like} of shape (n_channels)
        Vector with nodes to be selected.

    Returns
    -------
    ch_size : list
        Mean sizes for each channel.
    """
    if effect_size:
        # Multiply each t-val by effect size
        dt_nsubs = df["nsub"].unique()
        weights = np.sqrt(list(dt_nsubs))
        weights = weights / weights.sum()

        for dataset, w in zip(df.dataset.unique(), weights):
            df.loc[df.dataset == dataset, "t_val"] = (
                w * df.loc[df.dataset == dataset, "t_val"]
            )
        ch_t_val = df.groupby("node")["t_val"].apply(list).to_dict()
        ch_size_ = [
            np.mean(ch_t_val[ch]) for ch in channel_names
        ]  # this is the sum including size effect
    else:
        # Compute mean across t-values for each channel
        ch_t_val = df.groupby("node")["t_val"].apply(list).to_dict()
        ch_size_ = [np.mean(ch_t_val[ch]) for ch in channel_names]

    return np.array(ch_size_)


def combine_channel_pvalues(df, channels):
    """
    Combines p-values for each channel in two steps:
    1. Combine across subjects within each dataset (per channel).
    2. Combine across datasets (per channel), handling missing channels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'node', 'dataset', 'p_val', 'nsub'.
    channels : list of str
        List of channel names to process.

    Returns
    -------
    np.ndarray
        Array of combined p-values, one per channel.
    """
    # Step 1: Combine subject-level p-values within each dataset for each channel
    within_dataset_pvals = {}  # (dataset, channel) -> combined pval
    dataset_weights = {}  # (dataset, channel) -> nsub (used as weight)

    for (dataset, ch), df_grp in df.groupby(["dataset", "node"]):
        pvals = df_grp["p_val"].values
        nsub = df_grp["nsub"].iloc[0]
        if len(pvals) > 1:
            w = np.ones_like(pvals)
            _, p_comb = combine_pvalues(pvals, method="stouffer", weights=w)
        else:
            p_comb = pvals[0]
        within_dataset_pvals[(dataset, ch)] = p_comb
        dataset_weights[(dataset, ch)] = nsub

    # Step 2: Combine dataset-level p-values for each channel
    final_pvals = {}
    for ch in channels:
        pvals, weights = [], []
        for dataset in df["dataset"].unique():
            key = (dataset, ch)
            if key in within_dataset_pvals:
                pvals.append(within_dataset_pvals[key])
                weights.append(np.sqrt(dataset_weights[key]))

        if pvals:
            _, p_comb = combine_pvalues(pvals, method="stouffer", weights=weights)
        else:
            p_comb = 1.0  # If channel missing from all datasets
        final_pvals[ch] = p_comb

    return np.array([final_pvals.get(ch, 1.0) for ch in channels])


# Load parameters
params, paradigm = load_config()

# Get electrodes positions
ch_keys = np.unique(sum(params["ch_names"].values(), []))
positions = channel_pos(ch_keys, dimension="2d")
pos = {key: pos for key, pos in zip(ch_keys, positions)}

# Load stat data
df_t_test = pd.read_csv(ConfigPath.RES_DIR / "stats/t_test_perm5000.csv")

# Methods
methods = {metric: "netmetric" for name, metric in params["net_metrics"].items()}
methods["psdwelch"] = "psdwelch"

# Add number of subjects per dataset
df_t_test["nsub"] = None
for dt in df_t_test.dataset.unique():
    df_dt = df_t_test[df_t_test.dataset == dt]
    nsubs = df_dt.subject.unique().shape[0]
    df_t_test.loc[df_t_test.dataset == dt, "nsub"] = nsubs

for metric, method in methods.items():
    # Filter by selected metric and create dict of channels and t-values
    df_metric = df_t_test[df_t_test["metric"] == metric]

    # Define colors
    palette = (
        params["colorbar"]["psd"]
        if method == "psdwelch"
        else params["colorbar"]["net"] if method == "netmetric" else None
    )

    print("*" * 100)
    # for each dataset or all datasets together
    for dts in np.append(df_metric.dataset.unique(), "all"):
        print(f"{metric}: {dts}")
        if dts == "all":
            df_metric_dt = df_metric.loc[:]
            ch_names_dts = ch_keys
        else:
            df_metric_dt = df_metric.loc[df_metric["dataset"] == dts]
            ch_names_dts = params["ch_names"][dts]

        ch_names = df_metric_dt.node.unique()
        xlim_max = max(positions[:, 0])
        ch_names = ch_names[~np.isin(ch_names, EXCLUDE_CHANNELS)]

        # Compute t-values (mean channel size)
        ch_size = channel_size(df_metric_dt, ch_names, effect_size=False)

        # Combine p-values per channels
        combined_pvals = combine_channel_pvalues(df_metric_dt, ch_names)

        # Apply Bonferroni correction
        rejected, pvals_corrected, _, _ = multipletests(
            combined_pvals, alpha=P_VAL, method="bonferroni"
        )

        # Determine indices of significant or top 10 channels
        if np.any(rejected):
            ch_name_idx = np.where(rejected)[0]
            msg = "Significant channels (Bonferroni corrected):"
        else:
            ch_name_idx = np.argsort(np.abs(ch_size))[-10:]
            msg = "No significant channels after Bonferroni correction. Showing top 10 t-values:"

        # Sort selected channels by absolute t-value
        sorted_idx = ch_name_idx[np.argsort(np.abs(ch_size[ch_name_idx]))]
        sig_channels_sort = ch_names[sorted_idx]
        print(f"{msg} {sig_channels_sort}")

        # Plot t-values
        fig, ax = feature_plot_2d(ch_size, ch_names, palette=palette)
        plt.show()
        fig_name = ConfigPath.RES_DIR / f"stats/t_test_{metric}_{dts}.png"
        fig.savefig(fig_name, transparent=True)

        # Get 3D layout and save
        save_mat_file(
            ch_size,
            palette,
            ch_names,
            f"t_test_{metric}_{dts}",
            ch_name_idx=ch_name_idx,
        )
