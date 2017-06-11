import numpy as np
import sklearn.metrics as metrics
from core.metric.symmetric_distance_computer import SymmetricDistanceComputer
from core.metric.asymmetric_distance_computer import AsymmetricDistanceComputer
from . import py_nearest_indices_searcher
from core.common.timer import Timer

class NearestSearch:
    def find_nearest_indices(self, X: np.ndarray, Q: np.ndarray, n_nearest=None, metric='l2', n_jobs=1,
                             return_distances=False):
        Q = np.atleast_2d(Q)

        if n_nearest is None:
            n_nearest = len(X)

        # with Timer() as timer:
        #     print('start distances')
        vectors_distances_matrix = self.find_vectors_distances(X, Q, metric, n_jobs)
            # print('end distances')
            # print('start sort')
        nearest_indices = self.find_nearest_indices_(vectors_distances_matrix, n_nearest)
            # print('end sort\n')

        if return_distances:
            return (nearest_indices, vectors_distances_matrix)
        else:
            return nearest_indices

    def find_nearest_indices_old(self, X: np.ndarray, Q: np.ndarray, n_nearest=None, metric='l2', n_jobs=1,
                                 return_distances=False, partition_only=False):
        Q = np.atleast_2d(Q)

        vectors_distances_matrix = self.find_vectors_distances(X, Q, metric, n_jobs)
        if (not n_nearest):
            n_nearest = vectors_distances_matrix.shape[1]

        x_len = len(Q)
        rows_range = np.arange(x_len).reshape((-1, 1))
        nearest_indices = np.argpartition(vectors_distances_matrix, axis=1, kth=n_nearest - 1)[:, :n_nearest]
        if not partition_only:
            nearest_indices = nearest_indices[
                rows_range, np.argsort(vectors_distances_matrix[rows_range, nearest_indices])]

        if return_distances:
            return (nearest_indices, vectors_distances_matrix)
        else:
            return nearest_indices

    def find_nearest_indices_(self, vectors_distances_matrix: np.ndarray, n_nearest):
        n_nearest = int(n_nearest)
        # nearest_indices = np.argsort(vectors_distances_matrix, axis=1)[:, :n_nearest]
        n_base_vectors = vectors_distances_matrix.shape[1]
        py_nis = py_nearest_indices_searcher.PyNearestIndicesSearcher(n_base_vectors)
        vectors_distances_matrix=vectors_distances_matrix.astype(dtype='float32', order='C')
        nearest_indices = py_nis.findNearestIndices(vectors_distances_matrix, n_nearest)

        return nearest_indices

    def find_vectors_distances(self, X: np.ndarray, Q: np.ndarray, metric='l2', n_jobs=1):
        if isinstance(metric, SymmetricDistanceComputer):
            sdc = metric
            vectors_distances_matrix = sdc.computePairwiseSquaredDistances(Q, X)
        elif isinstance(metric,AsymmetricDistanceComputer):
            adc = metric
            vectors_distances_matrix = adc.computePairwiseSquaredDistances(Q, X)
        else:
            # print("X", X.shape, "Q", Q.shape)
            # X = X.squeeze()
            # Q = Q.squeeze()
            vectors_distances_matrix = metrics.pairwise_distances(Q, X, metric=metric, n_jobs=n_jobs)

        return vectors_distances_matrix
