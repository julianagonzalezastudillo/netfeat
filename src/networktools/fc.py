"""Estimation of functional connectivity."""
import numpy as np
import networktools.fc_mne as fc_mne
from sklearn.base import BaseEstimator, TransformerMixin


class FunctionalConnectivity(TransformerMixin, BaseEstimator):
    """
    Compute functional connectivity matrices.

    Parameters:
    -----------
    method : str, optional (default="coh")
        The method to compute functional connectivity. Options might include "coh" for coherence, "plv" for phase-locking value, etc.

    fmin : float, optional (default=8)
        The minimum frequency of the band of interest.

    fmax : float, optional (default=35)
        The maximum frequency of the band of interest.
    """

    def __init__(self, method="coh", fmin=8, fmax=35):
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
            Xfc_matrix = abs(
                fc_mne._compute_fc_subtrial(
                    X[i], delta, ratio, self.method, self.fmin, self.fmax
                )
            )
            np.fill_diagonal(Xfc_matrix, 0)
            Xfc[i, :, :] = Xfc_matrix

        return Xfc
