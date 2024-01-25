"""
=================================
            NETFEAT
=================================
Plot classification results. Here we plot the results using some of the convenience methods within the toolkit.
The score_plot visualizes all the data with one score per subject for every dataset and pipeline.
"""
import os
import pandas as pd
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
from moabb.analysis.meta_analysis import compute_dataset_statistics
import matplotlib.pyplot as plt
from plottools.plot_scores import score_plot, meta_analysis_plot
from config import ConfigPath, Info


# Define constants
PIPELINES = ["net", "csp", "riemannian"]

# Load datasets
datasets = [
    BNCI2014001(),
    Cho2017(),
    Lee2019_MI(),
    MunichMI(),
    PhysionetMI(),
    Shin2017A(accept=True),
    Schirrmeister2017(),
    Weibo2014(),
    Zhou2016(),
]

# Load results
results = pd.DataFrame()
for dt in datasets:
    for pipeline in PIPELINES:
        filename = ConfigPath.RES_CLASSIFY_DIR / f"{dt.code}_rh_lh_{pipeline}.csv"
        df_dt = pd.read_csv(filename)
        results = pd.concat([results, df_dt], ignore_index=True)

# Filter and consolidate results
select_pipelines = ["RG+SVM", "CSP+PS+SVM"]
select_results = results[results["pipeline"].isin(select_pipelines)]

for name, metric in Info.NET_METRICS.items():
    select = results[results["pipeline"] == f"coh+{name}+SVM"]
    select_results = pd.concat([select_results, select], ignore_index=True)

# Plot accuracies
fig, ax = score_plot(select_results)
plt.show()

# Plot stats
stats = compute_dataset_statistics(results)
for algo, algo_name in zip(
    ["RG+SVM", "coh+intg+SVM", "coh+lat+SVM", "coh+s+SVM", "coh+seg+SVM"],
    ["RG+SVM", "intg+SVM", "lat+SVM", "s+SVM", "seg+SVM"],
):
    fig, ax = meta_analysis_plot(stats, algo, "CSP+PS+SVM", algo_name, "CSP+SVM")
    plt.show()
# fig.savefig('test.png', transparent=True)
