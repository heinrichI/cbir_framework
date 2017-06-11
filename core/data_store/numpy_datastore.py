import itertools
from contextlib import ExitStack

import numpy as np
from core.data_store import datastore

from core.common.aggregate_iterable import aggregate_iterable


class NumpyDataStore(datastore.DataStore):
    # TODO decide about ids
    def __init__(self, items_sorted_by_ids: np.ndarray = None):
        self.save_items_sorted_by_ids(items_sorted_by_ids)

        # if ids_sorted is None:
        #     self.ids_sorted = np.arange(1, len(items_sorted_by_ids) + 1)
        # else:
        # self.ids_sorted = ids_sorted

    def get_items_sorted_by_ids(self, ids_sorted: np.ndarray = None):
        if ids_sorted is not None:
            ids_sorted = np.array(ids_sorted, dtype='int32')
            ids_sorted -= 1
            return np.take(self.items_sorted_by_ids, ids_sorted, axis=0)
        else:
            return self.items_sorted_by_ids

    def get_count(self):
        return len(self.items_sorted_by_ids)

    def save_items_sorted_by_ids(self, items_sorted_by_ids: np.ndarray, ids_sorted: np.ndarray = None):
        if items_sorted_by_ids is None:
            return
        if ids_sorted is None:
            if isinstance(items_sorted_by_ids, list):
                items_sorted_by_ids = np.asarray(items_sorted_by_ids)
                ids_sorted = np.arange(1, len(items_sorted_by_ids) + 1)
            elif not isinstance(items_sorted_by_ids, np.ndarray):
                items_sorted_by_ids, items_sorted_by_ids_copy_ = itertools.tee(items_sorted_by_ids)
                items_len = sum(1 for _ in items_sorted_by_ids_copy_)
                ids_sorted = np.arange(1, items_len + 1)
            else:
                ids_sorted = np.arange(1, len(items_sorted_by_ids) + 1)
        elif not isinstance(ids_sorted, np.ndarray):
            ids_sorted = aggregate_iterable(ids_sorted)

        if not isinstance(items_sorted_by_ids, np.ndarray):
            items_sorted_by_ids = aggregate_iterable(items_sorted_by_ids, detect_final_shape_by_first_elem=True,
                                                     n_elements=len(ids_sorted))

        self.items_sorted_by_ids = items_sorted_by_ids
        self.ids_sorted = ids_sorted

    def get_ids_sorted(self):
        return self.ids_sorted

    def is_stream_data_store(self):
        return False

    def already_exists(self):
        return False


def to_numpy_datastore(ds: datastore.DataStore):
    with ExitStack() as stack:
        if hasattr(ds, '__enter__'):
            stack.enter_context(ds)
        array_ds = NumpyDataStore(ds.get_items_sorted_by_ids())
        return array_ds
