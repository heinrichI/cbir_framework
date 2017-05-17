import numpy as np
from sklearn.cluster import KMeans
from quantization.quantizer import Quantizer
from quantization import py_multi_index_util as pymiu


class PQQuantizer(Quantizer):
    def __init__(self, n_clusters=256, n_quantizers=2, n_jobs=-1, precompute_distances='auto', n_init=10, max_iter=200):
        self.n_clusters = n_clusters
        self.n_quantizers = n_quantizers
        self.subquantizers = []
        self.n_jobs = n_jobs
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.max_iter = max_iter
        self.py_multi_index_util = pymiu.PyMultiIndexUtil(n_quantizers, n_clusters)

    def fit(self, X: np.ndarray):
        self.subvector_length = len(X[0]) // self.n_quantizers
        # print(self.subvector_length)
        X = X.reshape((len(X), self.n_quantizers, self.subvector_length))
        for i in range(self.n_quantizers):
            subvectors = X[:, i, :]
            # subvectors = np.copy(X[:, i, :], order='C')
            # print("subvectorsshape",subvectors.shape)
            kmeans = KMeans(n_clusters=self.n_clusters, precompute_distances=self.precompute_distances,
                            n_jobs=self.n_jobs, max_iter=self.max_iter, n_init=self.n_init,
                            verbose=True).fit(subvectors)
            self.subquantizers.append(kmeans)

    def get_cluster_centers(self):
        cluster_centers = np.array((
            [self.subquantizers[i].cluster_centers_ for i in range(self.n_quantizers)]
        ))
        return cluster_centers

    def predict(self, X: np.ndarray):
        """
            X - matrix, rows: vectors
            get cluster indices for vectors in X
            X: [
                [x00,x01],
                [x10,x11],
                ...
            ]
            returns:
            [
                i0,
                i1,
                ...
            ]
        """
        subspace_indices = self.predict_subspace_indices(X)
        indices = self.py_multi_index_util.flat_indices(subspace_indices)
        return indices

    def predict_subspace_indices(self, X):
        """
            X - matrix, rows: vectors
            get cluster indices for vectors in X
            X: [
                [x00,x01],
                [x10,x11],
                ...
            ]
            returns:
            [
                [u0, v0],
                [u1, v1],
                ...
            ]
        """
        centroids = np.empty(shape=(len(X), self.n_quantizers), dtype=np.int32)
        self.subvector_length = len(X[0]) // self.n_quantizers
        X = X.reshape((len(X), self.n_quantizers, self.subvector_length))
        for i in range(self.n_quantizers):
            subvectors = X[:, i, :]
            subquantizer = self.subquantizers[i]
            # print("subvectorsshape", subvectors.shape)
            centroid_indexes = subquantizer.predict(subvectors)
            centroids[:, i] = centroid_indexes

        # centroids = centroids
        return centroids


"""
 self.flatindex_multipliers = np.ones((n_quantizers))
        for i in range(n_quantizers - 2, -1, -1):
            self.flatindex_multipliers[i] = self.flatindex_multipliers[i + 1] * n_clusters

 subspace_indices = self.predict_subspace_indices(X)
        n = X.shape[0]
        flat_indices = np.empty(n)
        for i, subspaces_index in enumerate(subspace_indices):
            flatindex = 0
            for dim in range(len(self.flatindex_multipliers)):
                flatindex += subspaces_index[dim] * self.flatindex_multipliers[dim]
            flat_indices[i] = flatindex

        return flat_indices
"""
