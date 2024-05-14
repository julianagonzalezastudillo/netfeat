import os
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

import warnings
import moabb
from moabb.datasets import (BNCI2014001,
                            Cho2017,
                            Lee2019_MI,
                            MunichMI,
                            PhysionetMI,
                            Shin2017A,
                            Schirrmeister2017,
                            Weibo2014,
                            Zhou2016)

from moabb.paradigms import LeftRightImagery

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from netfeat_pipeline import WithinSessionEvaluation_netfeat

from riemanniantools.channelselection import ElectrodeSelection
import json
import pathlib


moabb.set_log_level("info")
warnings.filterwarnings("ignore")

path = pathlib.Path().resolve()
jsonfile = "params.json"
if jsonfile in os.listdir(path):
    jsonfullpath = os.path.join(path, jsonfile)
    with open(jsonfullpath) as jsonfile:
        parameterDict = json.load(jsonfile)
fmin = parameterDict['fmin']
fmax = parameterDict['fmax']

datasets = [BNCI2014001(), Cho2017(), Lee2019_MI(), MunichMI(), PhysionetMI(), Shin2017A(accept=True),
            Schirrmeister2017(), Weibo2014(), Zhou2016()]

pipeline = {"RG+SVM": make_pipeline(
    Covariances('oas'),
    ElectrodeSelection(nelec=10),
    TangentSpace(),
    SVC(kernel="linear"))}

paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)

for dt in datasets:
    cross_val = WithinSessionEvaluation_netfeat(
        datasets=[dt],
        paradigm=paradigm,
        overwrite=True,
        hdf5_path=None,
    )
    results = cross_val.process(pipeline)
    results.to_csv("./results/classification/{0}_rh_lh_riemannian.csv".format(dt.code), index=False)
