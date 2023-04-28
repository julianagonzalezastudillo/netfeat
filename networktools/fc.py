from pathlib import Path
import os
import sys
import pickle
import gzip
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import mne


class functional_connectivity(TransformerMixin, BaseEstimator):

    def __init__(self, method=None, filter=None, fmin=None, fmax=None, file_name=None, X=None, trails=None,
                 force_compute=False):
        self.method = method
        self.filter = filter
        self.fmin = fmin
        self.fmax = fmax
        self.X_data = None
        self.file_name = file_name
        self.X = X  # this are epochs
        self.trails = trails
        self.force_compute = force_compute

    def fit(self, X, y=None):
        return self

    def transform(self, X_cv):

        if self.force_compute == True:
            X_cv = mne.concatenate_epochs(X_cv)
            Xfc = self._functional_connectivity(self, X_cv)
        else:
            fc_path = self.file_name[0]
            fc_file = self.file_name[1]
            if os.path.exists(fc_file):
                print("Loading functional connectivity ...")
                with gzip.open(fc_file, "r") as f:
                    Xfc = pickle.load(f)
                Xfc = Xfc[self.method]

            else:
                # load data
                print("Computing functional connectivity ...")
                X = self.X  ##### this is ALL sessions
                if self.filter is not None:
                    X = self._filter(X)

                Xfc = self._functional_connectivity(self, X)

                if not os.path.exists(fc_path):
                    os.makedirs(fc_path)

                with gzip.open(fc_file, "w") as f:  # save
                    Xfc_save = {self.method: Xfc}
                    pickle.dump(Xfc_save, f)

        # return Xfc[self.trails]
        return Xfc

    def _functional_connectivity(self, X):
        n_chan = len(X.ch_names)
        delta, ratio = 1, 0.5
        Xfc = np.empty((len(X), n_chan, n_chan))
        for i in range(len(X)):
            Xfc_matrix = abs(fc._compute_fc_subtrial(
                X[i], delta, ratio, self.method, self.fmin, self.fmax
            ))
            np.fill_diagonal(Xfc_matrix, 0)
            Xfc[i, :, :] = Xfc_matrix

        return Xfc
