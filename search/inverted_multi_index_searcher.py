from search.inverted_multi_index import inverted_multi_index as imi
from search.inverted_multi_index import py_inverted_multi_index as pyimi
from search import searcher
import numpy as np


class InvertedMultiIndexSearcher(searcher.Searcher):
    index_entry_type = np.int32
    vector_type = np.float32

    def __init__(self, x: np.ndarray, x_ids: np.ndarray, cluster_centers: np.ndarray):
        # self.imi_searhcer = imi.PyInvertedMultiIndexSearcher(cluster_centers)
        # self.imi_searhcer.init_build_invertedMultiIndex(x, x_ids)
        x = x.astype(self.vector_type, copy=False)
        x = x.reshape((x.shape[0], x.shape[1]))
        x_ids = x_ids.astype(self.index_entry_type, copy=False)
        x_ids = x_ids.reshape((x_ids.shape[0],))
        cluster_centers = cluster_centers.astype(self.vector_type, copy=False)
        cluster_centers = cluster_centers.reshape(
            (cluster_centers.shape[0], cluster_centers.shape[1], cluster_centers.shape[2]))
        py_imi = pyimi.PyInvertedMultiIndexBuilder().buildInvertedMultiIndex(x_ids, x,
                                                                             cluster_centers)
        self.py_imi_searcher = pyimi.PyInvertedMultiIndexSearcher(py_imi, cluster_centers)

    def find_nearest_ids(self, Q: np.ndarray, n_nearest):
        # return self.imi_searhcer.find_candidates(Q, n_nearest)
        Q = Q.astype(self.vector_type, copy=False)
        Q = Q.reshape((Q.shape[0], Q.shape[1]))
        nearest_matrix = np.empty((len(Q), n_nearest), dtype=np.int32)
        for i in range(len(Q)):
            nearest_matrix[i, ...] = self.py_imi_searcher.findNearest(Q[i], n_nearest)

        return nearest_matrix
