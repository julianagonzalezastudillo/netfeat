"""
===========================================
  NETFEAT - Save CSP filters and patterns
===========================================
This module is design to save CSP filters and patterns for each subject
"""
# Authors: Juliana Gonzalez-Astudillo <julianagonzalezastudillo@gmail.com>
#
import copy as cp
import pickle
import gzip

import numpy as np

from mne.decoding import CSP
from moabb.paradigms import LeftRightImagery
from tools import suppress_stdout
from config import load_config, ConfigPath

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


# Load data
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
params = load_config()
paradigm = LeftRightImagery(fmin=params["fmin"], fmax=params["fmax"])  # for rh vs lh
csp_component_order = params["csp_component_order"]

for dt in datasets:
    for subject in dt.subject_list:
        # Get data for the current subject
        X, y, metadata = paradigm.get_data(
            dataset=dt, subjects=[subject], return_epochs=True
        )

        # Create a deep copy of the info object
        info = cp.deepcopy(X.info)

        # Loop over sessions for the current subject
        for session in np.unique(metadata.session):
            ix = metadata.session == session
            X_session = X.get_data()[ix]
            y_session = y[ix]

            # Create and fit CSP
            csp = CSP(
                n_components=params["csp_n_components"],
                reg=None,
                transform_into="average_power",
                component_order=params["csp_component_order"],
            )
            with suppress_stdout():
                csp.fit_transform(X_session, y_session)
            patterns = csp.patterns_.T
            filters = csp.filters_.T

            # Save filters and patterns
            csp_data = {
                "sub": subject,
                "code": dt.code,
                "info": X.info,
                "component_order": params["csp_component_order"],
                "session": metadata.session,
                "n_components": params["csp_n_components"],
                "filters": filters,
                "patterns": patterns,
            }
            csp_file = (
                ConfigPath.RES_CSP
                / f"csp_{dt.code}_sub{str(subject).zfill(3)}_{session}_{csp_component_order}.gz"
            )
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
