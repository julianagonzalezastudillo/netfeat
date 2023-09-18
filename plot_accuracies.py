##############################################################################
#%% Plot classification results
# ----------------
# Here we plot the results using some of the convenience methods within the
# toolkit.  The score_plot visualizes all the data with one score per subject
# for every dataset and pipeline.
import os
import pandas as pd
from moabb.datasets import (BNCI2014001,
                            Cho2017,
                            Lee2019_MI,
                            MunichMI,
                            PhysionetMI,
                            Shin2017A,
                            Schirrmeister2017,
                            Weibo2014,
                            Zhou2016)
import matplotlib.pyplot as plt
from plottools.plot_scores import score_plot

# Set paths
path = os.getcwd()
results_path = os.path.join(path, 'results', 'classification')

# Define constants
PIPELINES = ["net", "csp", "riemannian"]

# Load datasets
datasets = [BNCI2014001(), Cho2017(), Lee2019_MI(), MunichMI(), PhysionetMI(), Shin2017A(accept=True),
            Schirrmeister2017(), Weibo2014(), Zhou2016()]

# Load results
results = pd.DataFrame()
for dt in datasets:
    for pipeline in PIPELINES:
        filename = os.path.join(results_path, f"{dt.code}_rh_lh_{pipeline}.csv")
        df_dt = pd.read_csv(filename)
        results = results.append(df_dt, ignore_index=True)

# Filter and consolidate results
select_pipelines = ["RG+SVM", "CSP+PS+SVM"]
select_results = results[results['pipeline'].isin(select_pipelines)]

net_metrics = {
    's': 'strength',
    'lat': 'local_laterality',
    'seg': 'segregation',
    'intg': 'integration',
}
for name, metric in net_metrics.items():
    select = results[results['pipeline'] == f"coh+{name}+SVM"]
    select_results = select_results.append(select, ignore_index=True)

# Plot the results
fig, ax = score_plot(select_results)
plt.show()
