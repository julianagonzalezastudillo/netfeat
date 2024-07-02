"""Estimation of power spectrum density."""
from sklearn.base import BaseEstimator, TransformerMixin
from mne.time_frequency import psd_array_welch
import numpy as np


class PSDWelch(BaseEstimator, TransformerMixin):
    """
    Compute power spectrum density using Welch's method.

    Parameters:
    -----------
    fmin : float, optional (default=8)
        The minimum frequency of the band of interest.

    fmax : float, optional (default=35)
        The maximum frequency of the band of interest.

    sfreq : float
        The sampling frequency.

    n_overlap : int
        The number of points of overlap between segments. Will be adjusted
        to be <= n_per_seg. The default value is 0.

    n_per_seg : int | None
        Length of each Welch segment (windowed with a Hamming window). Defaults
        to None, which sets n_per_seg equal to n_fft.
    """

    def __init__(self, fmin=8, fmax=35, n_overlap=0, sfreq=None, n_per_seg=None):
        self.fmin = fmin
        self.fmax = fmax
        self.n_overlap = n_overlap
        self.sfreq = sfreq
        self.n_per_seg = n_per_seg

    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X, y=None):
        # Apply the PSD transformation
        psd, freqs = psd_array_welch(
            X._data,
            sfreq=self.sfreq,
            fmin=self.fmin,
            fmax=self.fmax,
            n_fft=int(self.sfreq),
            n_overlap=int(self.sfreq // 2),
            n_per_seg=int(self.sfreq),
        )

        # compute features (mean logarithmic band power)
        psd_log = 10 * np.log10(psd)
        psd_mean = np.mean(psd_log, axis=2)
        return psd_mean
