"""
================================================
 3. NETFEAT - CSP
================================================

Processing pipeline using CSP filter and SVM.
"""

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from mne.decoding import CSP
import warnings
import moabb

from netfeat_pipeline import WithinSessionEvaluation_netfeat
from config import load_config, ConfigPath, DATASETS


moabb.set_log_level("info")
warnings.filterwarnings("ignore")
params, paradigm = load_config()

pipeline = {
    "CSP+PS+SVM": make_pipeline(
        CSP(
            n_components=params["csp_n_components"],
            reg=None,
            transform_into="average_power",
            component_order=params["csp_component_order"],
        ),
        SVC(kernel="linear"),
    )
}

for dt in DATASETS:
    cross_val = WithinSessionEvaluation_netfeat(
        datasets=[dt],
        paradigm=paradigm,
        overwrite=True,
        hdf5_path=None,
    )
    results = cross_val.process(pipeline)
    results.to_csv(
        ConfigPath.RES_CLASSIFY_DIR / f"{dt.code}_rh_lh_csp.csv", index=False
    )
