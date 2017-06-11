import numpy as np
from core.search.searcher import Searcher

from core.search import nearest_search
from core.data_store.datastore import DataStore


class ExhaustiveSearcher(Searcher):
    def __init__(self, items_ndarray: np.ndarray, ids_ndarray: np.ndarray, metric='l2', preprocess_Q_func=None):
        self.ids_ndarray = ids_ndarray
        self.nearest_search = nearest_search.NearestSearch()
        self.metric = metric
        self.preprocess_Q_func = preprocess_Q_func
        self.items_ndarray = items_ndarray.reshape(items_ndarray.shape[0], items_ndarray.shape[1])

    def find_nearest_ids(self, Q: np.ndarray, n_nearest=None):
        # if hasattr(self.metric, 'preprocess_Q'):
        #     Q = self.metric.preprocess_Q(Q)

        Q = Q.reshape((Q.shape[0], Q.shape[1]))
        nearest_indices_ndarray = self.nearest_search.find_nearest_indices(self.items_ndarray, Q, n_nearest=n_nearest,
                                                                           metric=self.metric)
        nearest_ids_ndarray = np.take(self.ids_ndarray, nearest_indices_ndarray)
        return nearest_ids_ndarray

