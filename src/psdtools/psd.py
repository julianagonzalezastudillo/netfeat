from sklearn.base import BaseEstimator, TransformerMixin
from mne.time_frequency import psd_array_welch
import numpy as np


class PSDWelch(BaseEstimator, TransformerMixin):
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
            X,
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
