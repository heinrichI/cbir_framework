from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters
from core.data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore
from contextlib import ExitStack
import numpy as np
import math
from . import py_symmetric_distance_computer


class SymmetricDistanceComputer:
    def __init__(self, pq_quantizer: PQQuantizer, centroids_pairwise_distances: np.ndarray):
        self.pq_quantizer = pq_quantizer
        k = pq_quantizer.n_clusters
        self.centroids_pairwise_distances = centroids_pairwise_distances.astype(dtype='float32', order='C')
        self.m = list(range(len(self.centroids_pairwise_distances)))
        self.distances_calculated = 0

    def __call__(self, q_codes: np.ndarray, x_codes: np.ndarray, *args, **kwargs):
        # q_codes = q_codes.ravel()
        # x_codes = x_codes.ravel()
        # q_codes = self.pq_quantizer.predict_subspace_indices(q).T.ravel()
        distance = 0
        distances = self.centroids_pairwise_distances[q_codes.astype(dtype=int), x_codes.astype(dtype=int)]
        distance = distances.sum()
        # for i in self.m:
        # sklearn pairwise_distances transforms vectors to float type, so need cast to int
        # distance += self.clusters_pairwise_distances[i][int(q_codes[i]), int(x_codes[i])]
        distance_sqrt = math.sqrt(distance)
        self.distances_calculated += 1
        print(self.distances_calculated)
        return distance_sqrt

    def computePairwiseSquaredDistances(self, Q: np.ndarray, X_codes: np.ndarray):
        Q_codes = self.preprocess_Q(Q)
        py_sdc = py_symmetric_distance_computer.PySymmetricDistanceComputer(X_codes, self.centroids_pairwise_distances)
        pairwise_squared_distances = py_sdc.computePairwiseSquaredDistances(Q_codes)
        return pairwise_squared_distances

    def preprocess_Q(self, Q):
        Q_codes_T = self.pq_quantizer.predict_subspace_indices_T(Q)
        Q_codes = Q_codes_T.astype(dtype=Q_codes_T.dtype, order='C')
        return Q_codes
