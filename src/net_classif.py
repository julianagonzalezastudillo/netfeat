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

from netfeat_pipeline import WithinSessionEvaluation_netfeat
from networktools.fc import FunctionalConnectivity
from networktools.net import NetMetric
from networktools.channelselection import NetSelection


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
montages = parameterDict['montage']
fc = "coh"

# for rh vs lh
datasets = [BNCI2014001(), Cho2017(), Lee2019_MI(), MunichMI(), PhysionetMI(), Shin2017A(accept=True),
            Schirrmeister2017(), Weibo2014(), Zhou2016()]

network_metrics = {'s': 'strength',
                   'lat': 'local_laterality',
                   'seg': 'segregation',
                   'intg': 'integration',
                   }

pipeline = {}  # Initialize an empty dictionary
for name, metric in network_metrics.items():
    pipeline["{0}+{1}+SVM".format(fc, name)] = make_pipeline(
        FunctionalConnectivity(method=fc, fmin=fmin, fmax=fmax),
        NetMetric(method=metric),
        NetSelection(),
        SVC(kernel='linear')
    )

paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)  # for rh vs lh

for dt in datasets:
    dt.montage = montages[dt.code]

for dt in datasets:
    cross_val = WithinSessionEvaluation_netfeat(
        datasets=[dt],
        paradigm=paradigm,
        overwrite=True,
        hdf5_path=None,
        )
    results = cross_val.process(pipeline)
    results.to_csv("./results/classification/{0}_rh_lh_net.csv".format(dt.code), index=False)

