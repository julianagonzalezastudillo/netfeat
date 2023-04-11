from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from mne.decoding import CSP
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


# for rh vs lh
# datasets = [BNCI2014001(), Cho2017(), Lee2019_MI(), Schirrmeister2017(), Weibo2014(), Zhou2016()]
# montages = ['standard_1020', 'standard_1020', 'standard_1005', 'standard_1005', 'standard_1020', 'standard_1020']
datasets = [BNCI2014001()]
montages = ['standard_1020']

fmin = 8
fmax = 35
csp_component_order = 'mutual_info'
# csp_component_order = 'alternate'
csp_n_components = 8

pipeline = {"CSP+PS+SVM": make_pipeline(
    CSP(n_components=csp_n_components, reg=None, transform_into='average_power', component_order=csp_component_order),
    SVC(kernel = "linear"))}
paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)  # for rh vs lh

for dt, mtg in zip(datasets, montages):
    dt.montage = mtg
    cross_val = WithinSessionEvaluation_netfeat(
        datasets = dt,
        paradigm = paradigm,
        overwrite=True,
        hdf5_path=None,
    )
    results = cross_val.process(pipeline)
    results.to_csv("./results/classification/{0}_rh_lh_csp.csv".format(dt.code), index=False)

