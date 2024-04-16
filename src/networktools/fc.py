import numpy as np
import networktools.fc_mne as fc_mne
from sklearn.base import BaseEstimator, TransformerMixin


class FunctionalConnectivity(TransformerMixin, BaseEstimator):

    def __init__(self, method='coh', fmin=8, fmax=35):
        self.method = method
        self.fmin = fmin
        self.fmax = fmax

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xfc = self._functional_connectivity(X)
        return Xfc

    def _functional_connectivity(self, X):
        n_chan = len(X.ch_names)
        delta, ratio = 1, 0.5
        Xfc = np.empty((len(X), n_chan, n_chan))
        for i in range(len(X)):
            Xfc_matrix = abs(fc_mne._compute_fc_subtrial(
                X[i], delta, ratio, self.method, self.fmin, self.fmax
            ))
            np.fill_diagonal(Xfc_matrix, 0)
            Xfc[i, :, :] = Xfc_matrix

        return Xfc
