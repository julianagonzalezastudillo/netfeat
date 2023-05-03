import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
import networktools.net_spatial as net_spatial


class net_metric(TransformerMixin, BaseEstimator):

    def __init__(self, method='strength', montage_name='standard_1005'):
        self.method = method
        self.montage_name = montage_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xnet = self._net_metric(X, method=self.method)
        # Xnet = self.check_lateralization(np.array(metric_matrix))
        # Xnet = np.concatenate([Xnet, X], axis=1)  # if more than metric
        return Xnet

    def check_lateralization(self, X):
        net_metric_lateralization = ['local_laterality',
                                     'spatial_local_laterality',
                                     'segregation',
                                     'spatial_segregation',
                                     'integration',
                                     'spatial_integration'
                                     'inter_hemispheric',
                                     'inter_hemispheric_XC_LR'
                                     ]
        if self.method in net_metric_lateralization:
            X = X[:, :int(np.shape(X)[1]/2)]
        return X

    def _net_metric(self, X, method):
        ch_names = X.ch_names
        Xfc = X._data
        n_chan = np.size(Xfc, axis=1)
        ch_pos = net_spatial.positions_matrix(self.montage_name, X.ch_names)

        if method == 'strength':
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                G = nx.from_numpy_matrix(Xfc[i, :, :])  # to nx format
                metric_matrix[i, :] = [v for k, v in G.degree(weight = 'weight')]

        elif method == 'spatial_strength':
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                DIJ = net_spatial.distances_matrix(Xfc[i, :, :],
                                                    ch_pos)  # check electrodes positions are the same in both arrays
                metric_matrix[i, :] = net_spatial.spatial_strength(Xfc[i, :, :], DIJ)

        elif method == 'local_efficiency':
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                metric_matrix[i, :] = bct.efficiency_wei(Xfc[i, :, :], local = True)

        elif method == 'spatial_local_efficiency':
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                DIJ = net_spatial.distances_matrix(Xfc[i, :, :], ch_pos)
                XD = (Xfc[i, :, :] * DIJ) ** 2
                metric_matrix[i, :] = bct.efficiency_wei(XD, local = True)

        elif method == 'closeness_centrality':
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                G = nx.from_numpy_matrix(Xfc[i, :, :])
                metric_matrix[i, :] = list(topo_net.closeness_centrality_weighted(G).values())

        elif method == 'local_laterality':
            rh_idx, lh_idx, ch_idx, ch_bis_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                metric_matrix[i, :] = topo_net.local_laterality(Xfc[i, :, :], ch_names, ch_pos)

        elif method == 'spatial_local_laterality':
            rh_idx, lh_idx, ch_idx, ch_bis_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                DIJ = net_spatial.distances_matrix(Xfc[i, :, :],
                                                    ch_pos)  # check electrodes positions are the same in both arrays
                metric_matrix[i, :] = net_spatial.spatial_local_laterality(Xfc[i, :, :], DIJ, ch_names, ch_pos)

        elif method == 'local_laterality_all':
            rh_idx, lh_idx, ch_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                metric_matrix[i, :] = topo_net.local_laterality_all(Xfc[i, :, :], ch_names, ch_pos)

        elif method == 'spatial_local_laterality_all':
            rh_idx, lh_idx, ch_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                DIJ = net_spatial.distances_matrix(Xfc[i, :, :],
                                                    ch_pos)  # check electrodes positions are the same in both arrays
                metric_matrix[i, :] = net_spatial.spatial_local_laterality_all(Xfc[i, :, :], DIJ, ch_names, ch_pos)

        elif method == 'segregation':
            rh_idx, lh_idx, ch_idx, ch_bis_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                metric_matrix[i, :] = topo_net.segregation(Xfc[i, :, :], ch_names, ch_pos)

        elif method == 'seg2':
            rh_idx, lh_idx, ch_idx, ch_bis_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                metric_matrix[i, :] = topo_net.seg2(Xfc[i, :, :], ch_names, ch_pos)

        elif method == 'spatial_segregation':
            rh_idx, lh_idx, ch_idx, ch_bis_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                DIJ = net_spatial.distances_matrix(Xfc[i, :, :],
                                                    ch_pos)  # check electrodes positions are the same in both arrays
                metric_matrix[i, :] = net_spatial.spatial_segregation(Xfc[i, :, :], DIJ, ch_names, ch_pos)

        elif method == 'integration':
            rh_idx, lh_idx, ch_idx, ch_bis_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                metric_matrix[i, :] = topo_net.integration(Xfc[i, :, :], ch_names, ch_pos)

        elif method == 'spatial_integration':
            rh_idx, lh_idx, ch_idx, ch_bis_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                DIJ = net_spatial.distances_matrix(Xfc[i, :, :], ch_pos)  # check electrodes positions are the same in both arrays
                metric_matrix[i, :] = net_spatial.spatial_integration(Xfc[i, :, :], DIJ, ch_names, ch_pos)

        elif method == 'inter_hemispheric':
            rh_idx, lh_idx, ch_idx, ch_bis_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                metric_matrix[i, :] = topo_net.inter_hemispheric(Xfc[i, :, :], ch_names, ch_pos)

        elif method == 'inter_hemispheric_XC_LR':
            rh_idx, lh_idx, ch_idx, ch_bis_idx = topo_net.channel_idx(ch_names, ch_pos)
            n_chan = len(rh_idx) + len(lh_idx)
            metric_matrix = np.zeros((len(Xfc), n_chan), dtype = float)
            for i in range(len(Xfc)):
                metric_matrix[i, :] = topo_net.inter_hemispheric_XC_LR(Xfc[i, :, :], ch_names, ch_pos)

        elif method == 'global_laterality':
            metric_matrix = np.zeros((len(Xfc)), dtype = float)
            for i in range(len(Xfc)):
                metric_matrix[i] = topo_net.global_laterality(Xfc[i, :, :], ch_names, ch_pos)
            metric_matrix = metric_matrix.reshape(-1, 1)

        elif method == 'global_efficiency':
            metric_matrix = np.zeros((len(Xfc)), dtype=float)
            for i in range(len(Xfc)):
                metric_matrix[i] = bct.efficiency_wei(Xfc[i, :, :], local=False)
            metric_matrix = metric_matrix.reshape(-1, 1)

        elif method is None:
            sys.exit('No network metric was specified')
            # raise ValueError('')

        return metric_matrix
