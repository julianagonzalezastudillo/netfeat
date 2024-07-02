"""
=================================
 4. NETFEAT - Riemannian
=================================

Processing pipeline using Riemannian distance for feature selection + projection onto the tangent space and SVM.
"""
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import warnings
import moabb

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from netfeat_pipeline import WithinSessionEvaluationNetfeat
from config import load_config, ConfigPath, DATASETS

from riemanniantools.channelselection import ElectrodeSelection


moabb.set_log_level("info")
warnings.filterwarnings("ignore")
params, paradigm = load_config()

pipeline = {
    "RG+SVM": make_pipeline(
        Covariances("oas"),
        ElectrodeSelection(nelec=params["nelec"]),
        TangentSpace(),
        SVC(kernel="linear"),
    )
}

for dt in DATASETS:
    cross_val = WithinSessionEvaluationNetfeat(
        datasets=[dt],
        paradigm=paradigm,
        overwrite=True,
        hdf5_path=None,
    )
    results = cross_val.process(pipeline)
    results.to_csv(
        ConfigPath.RES_CLASSIFY_DIR / f"{dt.code}_rh_lh_riemannian.csv", index=False
    )
