import numpy as np
from sklearn.cluster import KMeans

from quantization.quantizer import Quantizer


class PQQuantizer(Quantizer):
    def __init__(self, n_clusters=256, n_quantizers=2, n_jobs=-1, precompute_distances='auto', n_init=10, max_iter=200):
        self.n_clusters = n_clusters
        self.n_quantizers = n_quantizers
        self.subquantizers = []
        self.n_jobs = n_jobs
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.max_iter = max_iter

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

    def predict(self, X):
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
