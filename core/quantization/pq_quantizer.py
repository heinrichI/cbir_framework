import numpy as np
from core.quantization import py_multi_index_util as pymiu
from sklearn.cluster import KMeans
import multiprocessing
from  core.common.measure_execution_time import mesure_time_wrapper
from core.common.file_utils import make_if_not_exists

from core.quantization.quantizer import Quantizer
import re


class PQQuantizer(Quantizer):
    def __init__(self, n_clusters=256, n_quantizers=2, allowed_time_minutes=5, **kmeans_kwargs):
        self.n_clusters = n_clusters
        self.n_quantizers = n_quantizers
        self.subquantizers = []
        self.py_multi_index_util = pymiu.PyMultiIndexUtil(n_quantizers, n_clusters)
        self.max_scalar_index = (n_clusters)**n_quantizers-1
        self.kmeans_kwargs = kmeans_kwargs
        # cpu_count = multiprocessing.cpu_count()
        self.kmeans_kwargs.setdefault('n_jobs', 2)
        self.kmeans_kwargs.setdefault('precompute_distances', 'auto')
        self.kmeans_kwargs.setdefault('n_init', 4)
        self.kmeans_kwargs.setdefault('max_iter', 30)
        self.kmeans_kwargs.setdefault('verbose', True)

    def modify_params(self, X: np.ndarray):
        if self.allowed_time != -1:
            wr = mesure_time_wrapper(self.fit_)
            test_kmeans_args = dict(self.kmeans_kwargs)
            test_kmeans_args['n_jobs'] = 1
            elapsed_time_minutes = wr(X, ) / 60

    def fit(self, X: np.ndarray):
        # self.modify_params(X)
        self.fit_(X, **self.kmeans_kwargs)

    def fit_(self, X: np.ndarray, **kmeans_kwargs):
        subvector_length = len(X[0]) // self.n_quantizers
        # print(subvector_length)
        X = X.reshape((len(X), self.n_quantizers, subvector_length))

        kmeans_kwargs = dict(self.kmeans_kwargs)
        kmeans_kwargs.setdefault('tol', float(0.001 * X.shape[2]))

        self.subquantizers = []
        self.quantization_info = []

        for i in range(self.n_quantizers):
            subvectors = X[:, i, :]
            # subvectors = np.copy(X[:, i, :], order='C')
            # print("subvectorsshape",subvectors.shape)
            kmeans = KMeans(n_clusters=self.n_clusters, **kmeans_kwargs)
            kmeans.fit(subvectors)
            self.subquantizers.append(kmeans)
            self.quantization_info.append(
                {'subspace': i, 'samples_dtype': str(X.dtype), 'subspace_samples_shape': str(subvectors.shape),
                 'kmeans_kwargs': kmeans_kwargs, 'inertia': float(kmeans.inertia_)})

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
        subspaced_indices = self.predict_subspace_indices(X)
        subspace_indices_ = np.transpose(subspaced_indices).astype(dtype=subspaced_indices.dtype, order='C')
        indices = self.py_multi_index_util.flat_indices(subspace_indices_)
        return indices

    def predict_subspace_indices(self, X) -> np.ndarray:
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
                [u0, u1, ..., u_len(X)],
                [v0, v1, ..., v_len(X)]
            ]
        """
        centroids = np.empty(shape=(self.n_quantizers, len(X)), dtype=np.int32)
        subvector_length = len(X[0]) // self.n_quantizers
        X = X.reshape((len(X), self.n_quantizers, subvector_length))
        for i in range(self.n_quantizers):
            subvectors = X[:, i, :]
            subquantizer = self.subquantizers[i]
            # print("subvectorsshape", subvectors.shape)
            centroid_indexes = subquantizer.predict(subvectors)
            centroids[i, :] = centroid_indexes

        # centroids = centroids
        return centroids

    def predict_subspace_indices_T(self, X) -> np.ndarray:
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
                [u0,v0],
                ...
                [u_len(X),v_len(X)]
            ]
        """
        centroids = self.predict_subspace_indices(X)
        centroids_T = centroids.T
        return centroids_T

    def get_quantization_info(self):
        import json
        quantization_info_josn = json.dumps(self.quantization_info)
        return quantization_info_josn


def restore_from_clusters(subspaced_clusters: np.ndarray) -> PQQuantizer:
    n_subspaces = subspaced_clusters.shape[0]
    n_clusters = subspaced_clusters.shape[1]
    # need to preserve order of clusters. Assuming n_init=1, max_iter=0, init=ndarray leads to such preserving
    pq_quantizer = PQQuantizer(n_clusters=n_clusters, n_quantizers=n_subspaces, n_init=1, max_iter=0)
    subvector_length = subspaced_clusters.shape[2]
    for i in range(n_subspaces):
        subclusters = subspaced_clusters[i]
        # print(subclusters.shape)
        kmeans = KMeans(n_clusters=n_clusters, precompute_distances='auto',
                        n_jobs=1, max_iter=1, n_init=1,
                        verbose=False, init=subclusters).fit(subclusters)
        pq_quantizer.subquantizers.append(kmeans)
    return pq_quantizer


def build_pq_params_str(pq_params: dict):
    pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])
    return pq_params_str


def extract_pq_params_from_str(pq_params_str):
    result = re.search(r'.*pq-([0-9]+)-([0-9]+).*', pq_params_str)
    # print(result.groups())
    try:
        k = int(result.group(1))
        m = int(result.group(2))
        pq_params = {'n_clusters': k, 'n_quantizers': m}
        return pq_params
    except:
        return None


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
