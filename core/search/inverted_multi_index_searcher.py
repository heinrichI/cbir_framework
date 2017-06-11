import numpy as np
from core.search import searcher

from core.search.inverted_multi_index import py_inverted_multi_index as pyimi


class InvertedMultiIndexSearcher(searcher.Searcher):
    index_entry_type = np.int32
    vector_type = np.float32

    def __init__(self, x_ids: np.ndarray, cluster_centers: np.ndarray, x: np.ndarray = None,
                 x_pqcodes: np.ndarray = None):
        """
        
        :param x_ids: 
        :param cluster_centers: 
        :param x: (excludes x_subspaced_centroid_indices)
        :param x_pqcodes:  (excludes x)
        """
        # self.imi_searhcer = imi.PyInvertedMultiIndexSearcher(cluster_centers)
        # self.imi_searhcer.init_build_invertedMultiIndex(x, x_ids)
        x_ids = x_ids.astype(self.index_entry_type, copy=False)
        x_ids = x_ids.reshape((x_ids.shape[0],))
        cluster_centers = cluster_centers.astype(self.vector_type, copy=False)
        cluster_centers = cluster_centers.reshape(
            (cluster_centers.shape[0], cluster_centers.shape[1], cluster_centers.shape[2]))

        if x is not None:
            x = x.astype(self.vector_type, copy=False)
            x = x.reshape((x.shape[0], x.shape[1]))
            py_imi = pyimi.PyInvertedMultiIndexBuilder().buildInvertedMultiIndexFromVectors(x_ids, x,
                                                                                            cluster_centers)
        else:
            x_pqcodes = x_pqcodes.astype(dtype='int32', copy=False)
            x_pqcodes = x_pqcodes.reshape(
                (x_pqcodes.shape[0], x_pqcodes.shape[1]))
            py_imi = pyimi.PyInvertedMultiIndexBuilder().buildInvertedMultiIndex(x_ids, x_pqcodes,
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


def calculate_possible_params(bytes_free, logK_from=3, logK_to=12, logM_from=1, logM_to=5):
    K_arr = [2 ** i for i in range(logK_from, logK_to)]
    m_arr = [2 ** i for i in range(logM_from, logM_to)]
    params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in K_arr for m in m_arr
                  if 4 * K ** m < bytes_free]
    return params_arr
