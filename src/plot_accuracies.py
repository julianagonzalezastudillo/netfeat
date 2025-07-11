"""
============================================
 5.1. NETFEAT - Plot classification results
============================================

Plot classification results. Here we plot the results using some of the
convenience methods within the toolkit. The score_plot visualizes all the
data with one score per subject for every dataset and pipeline.
"""

import pandas as pd
import numpy as np
from moabb.analysis.meta_analysis import compute_dataset_statistics
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plottools.plot_scores import score_plot, meta_analysis_plot
from config import ConfigPath, load_config, DATASETS


def format_mean_std(group):
    """Custom function to format mean and std together."""
    mean = group["mean"]
    std = group["std"]
    if np.isnan(std):
        return f"{mean:.1f}±NaN  |  "
    else:
        return f"{mean:.1f}±{std:.1f}  |  "


# Define constants
PIPELINES = ["net_t-test", "csp", "riemannian", "psd_t-test"]
params, paradigm = load_config()

# Load results
results = pd.DataFrame()
for dt in DATASETS:
    for pipeline in PIPELINES:
        filename = ConfigPath.RES_CLASSIFY_DIR / f"{dt.code}_rh_lh_{pipeline}.csv"
        df_dt = pd.read_csv(filename)
        results = pd.concat([results, df_dt], ignore_index=True)

# Filter and consolidate results
select_pipelines = ["PSD+SVM", "CSP+PS+SVM", "RG+SVM"]
select_results = results[results["pipeline"].isin(select_pipelines)]

for name, metric in params["net_metrics"].items():
    select = results[results["pipeline"] == f"coh+{name}+SVM"]
    select_results = pd.concat([select_results, select], ignore_index=True)

# Sort by pipeline order
select_results["pipeline"] = pd.Categorical(
    select_results["pipeline"], categories=params["pipeline_order"], ordered=True
)
select_results = select_results.sort_values("pipeline")

# Rename pipelines for plot
select_results["pipeline"] = select_results["pipeline"].replace(
    params["pipeline_names"]
)

# Plot accuracies and save
fig, ax = score_plot(select_results)
res_classify_plot = ConfigPath.RES_CLASSIFY_DIR / "plot"
res_classify_plot.mkdir(parents=True, exist_ok=True)
fig.savefig(res_classify_plot / "classif_acc.png", transparent=True)
plt.show()

# %% Plot stats
stats = compute_dataset_statistics(select_results)
stats = stats.sort_values(by="dataset", ascending=False)
algos = ["s+SVM", "λ+SVM", "σ+SVM", "ω+SVM"]
for algo in algos:
    fig, ax = meta_analysis_plot(stats, algo, "PSD+SVM")
    fig.savefig(res_classify_plot / f"classif_stats_{algo}.png", transparent=True)
    plt.show()

# Put four figure in one
images = [
    mpimg.imread(res_classify_plot / f"classif_stats_{algo}.png") for algo in algos
]
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=600)
[axs[i, j].imshow(images[i * 2 + j]) for i in range(2) for j in range(2)]

for ax_row in axs:
    for ax in ax_row:
        ax.axis("off")
plt.tight_layout()
fig.savefig(res_classify_plot / f"classif_stats.png", transparent=True)
plt.show()

# Print table of pipelines means
table = select_results
table["score"] *= 100

# Compute the mean and standard deviation of the scores per dataset and pipeline
stats_mean_std = table.groupby(["dataset", "pipeline"], observed=False)["score"].agg(
    ["mean", "std"]
)
stats_mean_std["mean_std"] = stats_mean_std.apply(format_mean_std, axis=1)
pivoted_stats_mean_std = stats_mean_std["mean_std"].unstack(level=1)
print(pivoted_stats_mean_std.reset_index().to_string(index=False))

# Print total means
resume_table = table.pivot_table(
    index="pipeline", columns="dataset", values="score", aggfunc="mean", observed=False
)
mean_across_datasets = resume_table.mean(axis=1)
std_across_datasets = resume_table.std(axis=1)
pipeline_stats_mean_std = pd.DataFrame(
    {"mean": mean_across_datasets, "std": std_across_datasets}
)
pipeline_stats_mean_std["mean_std"] = pipeline_stats_mean_std.apply(
    format_mean_std, axis=1
)
print(pipeline_stats_mean_std["mean_std"].to_string())
