"""
=================================
            NETFEAT
=================================

"""
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from mne.decoding import CSP
import warnings
import moabb
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
from moabb.paradigms import LeftRightImagery

from netfeat_pipeline import WithinSessionEvaluation_netfeat
from config import load_config


moabb.set_log_level("info")
warnings.filterwarnings("ignore")
params = load_config()

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
paradigm = LeftRightImagery(fmin=params["fmin"], fmax=params["fmax"])  # for rh vs lh

for dt in datasets:
    cross_val = WithinSessionEvaluation_netfeat(
        datasets=[dt],
        paradigm=paradigm,
        overwrite=True,
        hdf5_path=None,
    )
    results = cross_val.process(pipeline)
    results.to_csv(
        "./results/classification/{0}_rh_lh_csp.csv".format(dt.code), index=False
    )
