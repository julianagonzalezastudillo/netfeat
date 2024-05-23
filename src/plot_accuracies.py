"""
=================================
            NETFEAT
=================================
Plot classification results. Here we plot the results using some of the convenience methods within the toolkit.
The score_plot visualizes all the data with one score per subject for every dataset and pipeline.
"""
import pandas as pd
from moabb.analysis.meta_analysis import compute_dataset_statistics
import matplotlib.pyplot as plt
from plottools.plot_scores import score_plot, meta_analysis_plot
from config import ConfigPath, load_config, DATASETS


# Define constants
PIPELINES = ["net", "csp", "riemannian", "psd_"]
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

# Sort by custom order
pipeline_order = [
    "PSD+SVM",
    "CSP+PS+SVM",
    "RG+SVM",
    "coh+s+SVM",
    "coh+lat+SVM",
    "coh+seg+SVM",
    "coh+intg+SVM",
]
select_results["pipeline"] = pd.Categorical(
    select_results["pipeline"], categories=pipeline_order, ordered=True
)
select_results = select_results.sort_values("pipeline")

# Plot accuracies
fig, ax = score_plot(select_results)
plt.show()

# Plot stats
stats = compute_dataset_statistics(results)
for algo, algo_name in zip(
    ["RG+SVM", "coh+s+SVM", "coh+lat+SVM", "coh+seg+SVM", "coh+intg+SVM"],
    ["RG+SVM", "s+SVM", "lat+SVM", "seg+SVM", "intg+SVM"],
):
    fig, ax = meta_analysis_plot(stats, algo, "CSP+PS+SVM", algo_name, "CSP+SVM")
    plt.show()
# fig.savefig('test.png', transparent=True)
