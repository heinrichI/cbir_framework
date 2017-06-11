from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters
from core.data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore
from contextlib import ExitStack
import numpy as np
import math
from . import py_asymmetric_distance_computer


class AsymmetricDistanceComputer:
    def __init__(self, subspaced_centroids: np.ndarray):
        subspaced_centroids = subspaced_centroids.astype(dtype='float32', order='C')
        self.subspaced_centroids = subspaced_centroids

    def computePairwiseSquaredDistances(self, Q: np.ndarray, X_codes: np.ndarray):
        py_adc = py_asymmetric_distance_computer.PyAsymmetricDistanceComputer(X_codes, self.subspaced_centroids)
        Q = Q.astype(dtype='float32', order='C')
        pairwise_squared_distances = py_adc.computePairwiseSquaredDistances(Q)
        return pairwise_squared_distances
