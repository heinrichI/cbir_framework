import numpy as np

from search.searcher import Searcher
from search import nearest_search


class ExhaustiveSearcher(Searcher):
    def __init__(self, items_ndarray: np.ndarray, ids_ndarray: np.ndarray):
        self.items_ndarray = items_ndarray
        self.ids_ndarray = ids_ndarray
        self.nearest_search = nearest_search.NearestSearch()

    def find_nearest_ids(self, Q: np.ndarray, n_nearest=None, metric="l1"):
        nearest_indices_ndarray = self.nearest_search.find_nearest_indices(self.items_ndarray, Q, n_nearest=n_nearest,
                                                                           metric=metric)
        nearest_ids_ndarray = np.take(self.ids_ndarray, nearest_indices_ndarray)
        return nearest_ids_ndarray
