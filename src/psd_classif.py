"""
=================================
            NETFEAT
=================================

"""
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import numpy as np

import moabb
import warnings

from netfeat_pipeline import WithinSessionEvaluation_netfeat
from psdtools.psd import PSDWelch

from psdtools.channelselection import PSDSelection
from config import load_config, ConfigPath, DATASETS


moabb.set_log_level("info")
warnings.filterwarnings("ignore")
params, paradigm = load_config()

pipeline = {
    "PSD+SVM": make_pipeline(
        PSDWelch(
            fmin=params["fmin"],
            fmax=params["fmax"],
        ),
        PSDSelection(),
        SVC(kernel="linear"),
    )
}

for dt in DATASETS:
    dt.montage = params["montage"][dt.code]
    cross_val = WithinSessionEvaluation_netfeat(
        datasets=[dt],
        paradigm=paradigm,
        overwrite=True,
        hdf5_path=None,
    )
    results = cross_val.process(pipeline)

    results.to_csv(
        ConfigPath.RES_CLASSIFY_DIR / f"{dt.code}_rh_lh_psd_.csv", index=False
    )
    print(np.mean(results["score"]))
