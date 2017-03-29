import numpy as np

from data_store import datastore


class NumpyDataStore(datastore.DataStore):
    #TODO decide about ids
    def __init__(self, items_sorted_by_ids: np.ndarray=None):
        self.items_sorted_by_ids = items_sorted_by_ids
        # if ids_sorted is None:
        #     self.ids_sorted = np.arange(1, len(items_sorted_by_ids) + 1)
        # else:
        # self.ids_sorted = ids_sorted

    def get_items_sorted_by_ids(self, ids_sorted: np.ndarray = None):
        if ids_sorted is not None:
            return np.take(self.items_sorted_by_ids, ids_sorted, axis=0)
        else:
            return self.items_sorted_by_ids

    def get_count(self):
        return len(self.items_sorted_by_ids)

    def save_items_sorted_by_ids(self, items_sorted_by_ids: np.ndarray, ids_sorted: np.ndarray = None):
        self.items_sorted_by_ids = items_sorted_by_ids

    def get_ids_sorted(self):
        return None

    def is_stream_data_store(self):
        return False
