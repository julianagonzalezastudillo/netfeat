import numpy as np
import sys
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
import networktools.net_topo as net_topo


class NetMetric(TransformerMixin, BaseEstimator):
    def __init__(self, method="strength", montage_name="standard_1005", dataset=None):
        self.method = method
        self.montage_name = montage_name
        self.dataset = dataset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xnet = self._net_metric(X, method=self.method)
        # Xnet = self.check_lateralization(np.array(metric_matrix))
        # Xnet = np.concatenate([Xnet, X], axis=1)  # if more than metric
        return Xnet

    def _net_metric(self, X, method):
        ch_names = X.ch_names
        Xfc = X._data
        n_chan = np.size(Xfc, axis=1)

        if self.dataset.code == "Grosse-Wentrup 2009":
            rh_idx, lh_idx, ch_idx, ch_bis_idx = net_topo.channel_idx_MunichMI()
        else:
            ch_pos = net_topo.positions_matrix(self.montage_name, X.ch_names)
            rh_idx, lh_idx, ch_idx, ch_bis_idx = net_topo.channel_idx(ch_names, ch_pos)

        if method == "strength":
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype=float)
            for i in range(len(Xfc)):
                G = nx.from_numpy_array(Xfc[i, :, :])  # to nx format
                metric_matrix[i, :] = [v for k, v in G.degree(weight="weight")]

        elif method == "local_laterality":
            metric_matrix = np.zeros((len(Xfc), len(rh_idx) + len(lh_idx)), dtype=float)
            for i, fc_data in enumerate(Xfc):
                metric_matrix[i, :] = net_topo.local_laterality(
                    fc_data, lh_idx, rh_idx, ch_idx, ch_bis_idx
                )

        elif method == "integration":
            metric_matrix = np.zeros((len(Xfc), len(rh_idx) + len(lh_idx)), dtype=float)
            for i, fc_data in enumerate(Xfc):
                metric_matrix[i, :] = net_topo.integration(
                    fc_data, lh_idx, rh_idx, ch_idx, ch_bis_idx
                )

        elif method == "segregation":
            metric_matrix = np.zeros((len(Xfc), len(rh_idx) + len(lh_idx)), dtype=float)
            for i, fc_data in enumerate(Xfc):
                metric_matrix[i, :] = net_topo.segregation(
                    fc_data, lh_idx, rh_idx, ch_idx, ch_bis_idx
                )

        elif method is None:
            sys.exit("No network metric was specified")
            # raise ValueError('')

        return metric_matrix
