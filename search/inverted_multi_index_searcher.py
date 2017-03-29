from search.inverted_multi_index import inverted_multi_index as imi
from search import searcher
import numpy as np


class InvertedMultiIndexSearcher(searcher.Searcher):
    def __init__(self, x, x_ids, cluster_centers):
        self.imi_searhcer = imi.PyInvertedMultiIndexSearcher(cluster_centers)
        self.imi_searhcer.init_build_invertedMultiIndex(x, x_ids)

    def find_nearest_ids(self, Q: np.ndarray, n_nearest):
        return self.imi_searhcer.find_candidates(Q, n_nearest)
