"""Code for channel selection based on a Riemannian geometry criterion."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.utils.distance import distance
from pyriemann.classification import MDM

from operator import itemgetter
from src.moabb_settings import save_global


class ElectrodeSelection(BaseEstimator, TransformerMixin):
    """Channel selection based on a Riemannian geometry criterion.

    For each class, a centroid is estimated, and the channel selection is based
    on the maximization of the distance between centroids. This is done by a
    backward elimination where the electrode that carries the less distance is
    removed from the subset at each iteration.
    This algorith is described in [1].

    Parameters
    ----------
    nelec : int (default 16)
        the number of electrode to keep in the final subset.

    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the selection.

    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : list
        the class centroids.

    dist_ : list
        list of distance at each interation.

    References
    ----------
    [1] A. Barachant and S. Bonnet, "Channel selection procedure using
    riemannian distance for BCI applications," in 2011 5th International
    IEEE/EMBS Conference on Neural Engineering (NER), 2011, 348-351
    """

    def __init__(
        self,
        nelec=16,
        metric="riemann",
        n_jobs=1,
        dataset=None,
        session=None,
        sessions_name=None,
        ch_names=None,
        pipeline=None,
        cv_splits=None,
    ):
        """Init."""
        self.nelec = nelec
        self.metric = metric
        self.n_jobs = n_jobs
        self.dataset = dataset
        self.session = session
        self.sessions_name = sessions_name
        self.ch_names = ch_names
        self.pipeline = pipeline
        self.cv_splits = cv_splits
        self.t_val = None

    def fit(self, X, y=None, sample_weight=None):
        """Find the optimal subset of electrodes.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample. if None, each sample is treated with
            equal weights.

        Returns
        -------
        self : ElectrodeSelection instance
            The ElectrodeSelection instance.
        """
        mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)
        mdm.fit(X, y, sample_weight=sample_weight)
        self.covmeans_ = mdm.covmeans_

        Ne, _ = self.covmeans_[0].shape

        self.dist_ = []
        self.subelec_ = list(range(0, Ne, 1))
        while (len(self.subelec_)) > self.nelec:
            di = np.zeros((len(self.subelec_), 1))
            for idx in range(len(self.subelec_)):
                sub = self.subelec_[:]
                sub.pop(idx)
                di[idx] = 0
                for i in range(len(self.covmeans_)):
                    for j in range(i + 1, len(self.covmeans_)):
                        di[idx] += distance(
                            self.covmeans_[i][:, sub][sub, :],
                            self.covmeans_[j][:, sub][sub, :],
                            metric=mdm.metric_dist,
                        )
            # print di
            torm = di.argmax()
            self.dist_.append(di.max())
            self.subelec_.pop(torm)

        self.subelec_names = np.array(itemgetter(*self.subelec_)(self.ch_names))
        save_global(self)
        # print('call {0}'.format(np.shape(X)[0]))
        # print('cv: {0}, ch: {1}, size: {2}'.format(n_cv, self.subelec_names, np.shape(X)[0]))
        return self

    def transform(self, X):
        """Return reduced matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        covs : ndarray, shape (n_trials, n_elec, n_elec)
            The covariances matrices after reduction of the number of channels.
        """

        return X[:, self.subelec_, :][:, :, self.subelec_]
