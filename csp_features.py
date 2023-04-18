"""
=================================
  Save CSP filters and patterns
=================================
This module is design to save CSP filters and patterns for each subject
"""
# Authors: Juliana Gonzalez-Astudillo <julianagonzalezastudillo@gmail.com>
#
import numpy as np
from mne.decoding import CSP
from tools import suppress_stdout
from moabb.datasets import (BNCI2014001,
                            Cho2017,
                            Lee2019_MI,
                            MunichMI,
                            Shin2017A,
                            Schirrmeister2017,
                            Weibo2014,
                            Zhou2016)

from moabb.paradigms import LeftRightImagery
import copy as cp
import os
from pathlib import Path
import pickle
import gzip
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
n_components = parameterDict['csp_n_components']
component_order = parameterDict['csp_component_order']

# datasets = [BNCI2014001()]
datasets = [Shin2017A(accept=True), Schirrmeister2017(), Weibo2014(), Zhou2016()]
paradigm = LeftRightImagery(fmin=fmin, fmax=fmax)  # for rh vs lh

for dt in datasets:
    for subject in dt.subject_list:
        X, y, metadata = paradigm.get_data(dataset=dt, subjects=[subject], return_epochs=True)

        info = cp.deepcopy(X.info)
        for session in np.unique(metadata.session):
            ix = metadata.session == session
            X_session = X.get_data()[ix]
            y_session = y[ix]

            csp = CSP(n_components=n_components, reg=None, transform_into='average_power', component_order=component_order)
            with suppress_stdout():
                csp.fit_transform(X_session, y_session)
            patterns = csp.patterns_.T
            filters = csp.filters_.T

            # save filters and patterns
            csp_data = {'sub': subject,
                        'code': dt.code,
                        'info': X.info,
                        'component_order': component_order,
                        'session': metadata.session,
                        'n_components': n_components,
                        'filters': filters,
                        'patterns': patterns}
            csp_file = Path(os.path.join(os.getcwd(), 'results', 'csp_features', 'csp_{0}_sub{1}_{2}_{3}.gz'
                                         .format(dt.code, str(subject).zfill(3), session, component_order)))
            with gzip.open(csp_file, "w") as f:  # save
                pickle.dump(csp_data, f)

            # for testing
            # import matplotlib.pyplot as plt
            # import mne
            # csp.plot_patterns(X.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
            # mne.viz.plot_topomap(csp.patterns_.T[:, 0], info, show=False)
            # plt.show()
            # csp.plot_filters(X.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
            # mne.viz.plot_topomap(csp.filters_.T[:, 0], info, show=False)
            # plt.show()