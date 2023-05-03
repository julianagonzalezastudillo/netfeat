import os
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from moabb.datasets import (BNCI2014001,
                            BNCI2015001,
                            BNCI2015004,
                            Lee2019_MI,
                            Zhou2016,
                            MunichMI,
                            Shin2017A,
                            Cho2017,
                            Schirrmeister2017,
                            Weibo2014)

from moabb.paradigms import LeftRightImagery

from netfeat_pipeline import WithinSessionEvaluation_netfeat
from networktools.fc import functional_connectivity
from networktools.net import net_metric

import json
import pathlib


path = pathlib.Path().resolve()
jsonfile = "params.json"
if jsonfile in os.listdir(path):
    jsonfullpath = os.path.join(path, jsonfile)
    with open(jsonfullpath) as jsonfile:
        parameterDict = json.load(jsonfile)
fmin = parameterDict['fmin']
fmax = parameterDict['fmax']

# for rh vs lh
datasets = [Cho2017(), Lee2019_MI(), Schirrmeister2017(), Weibo2014(), Zhou2016()]
montages = ['standard_1020', 'standard_1005', 'standard_1005', 'standard_1020', 'standard_1020']
# datasets = [BNCI2014001()]
# montages = ['standard_1020']


network_metrics = {'s': 'strength',
                   # 'lat': 'local_laterality',
                   # 'seg': 'segregation',
                   # 'intg': 'integration',
                   }

for name, metric in network_metrics.items():
    pipeline = {"coh+{0}+SVM".format(name): make_pipeline(
        functional_connectivity(method="coh", fmin=fmin, fmax=fmax),
        net_metric(method=metric),
        SVC(kernel='linear'))}

paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)  # for rh vs lh

for dt, mtg in zip(datasets, montages):
    dt.montage = mtg
    cross_val = WithinSessionEvaluation_netfeat(
        datasets = dt,
        paradigm = paradigm,
        overwrite=True,
        hdf5_path=None,
    )
    results = cross_val.evaluate(dt, pipeline)
    results.to_csv("./results/classification/{0}_rh_lh_net.csv".format(dt.code), index=False)

