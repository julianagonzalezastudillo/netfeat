"""
=================================
 1. NETFEAT - Network properties
=================================

Processing pipeline using functional connectivity + network metrics and SVM.
"""

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import moabb

from netfeat_pipeline import WithinSessionEvaluation_netfeat
from networktools.fc import FunctionalConnectivity
from networktools.net import NetMetric
from preprocessing.channelselection import FeaturesSelection
from config import load_config, ConfigPath, DATASETS


moabb.set_log_level("info")
params, paradigm = load_config()
rank = "t-test"

pipeline = {}
for name, metric in params["net_metrics"].items():
    pipeline[f"{params['fc']}+{name}+SVM"] = make_pipeline(
        FunctionalConnectivity(
            method=params["fc"], fmin=params["fmin"], fmax=params["fmax"]
        ),
        NetMetric(method=metric),
        StandardScaler(),
        FeaturesSelection(rank=rank),
        SVC(kernel="linear"),
    )

for dt in DATASETS:
    cross_val = WithinSessionEvaluation_netfeat(
        datasets=[dt],
        paradigm=paradigm,
        overwrite=True,
        hdf5_path=None,
    )
    results = cross_val.process(pipeline)
    results.to_csv(
        ConfigPath.RES_CLASSIFY_DIR / f"{dt.code}_rh_lh_net_{rank}.csv", index=False
    )
