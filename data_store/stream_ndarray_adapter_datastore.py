import itertools
from contextlib import ExitStack
import numpy as np

from common.aggregate_iterable import aggregate_iterable
from data_store import datastore


class StreamNdarrayAdapterDataStore(datastore.DataStore):
    """
        adapts data_store that works with streams to work with ndarrays
    """

    def __init__(self, stream_data_store: datastore.DataStore):
        if not stream_data_store.is_stream_data_store():
            self.get_count = stream_data_store.get_count
            self.get_ids_sorted = stream_data_store.get_ids_sorted
            self.get_items_sorted_by_ids = stream_data_store.get_items_sorted_by_ids
            self.save_items_sorted_by_ids = stream_data_store.save_items_sorted_by_ids
        else:
            self.stream_data_store = stream_data_store

    def get_count(self):
        with ExitStack() as stack:
            if hasattr(self.stream_data_store, '__enter__'):
                stack.enter_context(self.stream_data_store)
            return self.stream_data_store.get_count()

    def get_items_sorted_by_ids(self, ids_sorted: np.ndarray = None):
        with ExitStack() as stack:
            if hasattr(self.stream_data_store, '__enter__'):
                stack.enter_context(self.stream_data_store)

            ids_sorted_stream = None
            if ids_sorted is not None:
                count_ = ids_sorted.shape[0]
                ids_sorted = ids_sorted.ravel()
                ids_sorted_stream = iter(ids_sorted)
            else:
                count_ = self.stream_data_store.get_count()

            items_sorted_by_ids_stream = self.stream_data_store.get_items_sorted_by_ids(ids_sorted_stream)
            items_sorted_by_ids_ndarray = aggregate_iterable(items_sorted_by_ids_stream, count_)

            return items_sorted_by_ids_ndarray

    def save_items_sorted_by_ids(self, items_sorted_by_ids: np.ndarray, ids_sorted: np.ndarray = None):
        with ExitStack() as stack:
            if hasattr(self.stream_data_store, '__enter__'):
                stack.enter_context(self.stream_data_store)

            ids_sorted_stream = None
            if ids_sorted is not None:
                ids_sorted = ids_sorted.ravel()
                ids_sorted_stream = iter(ids_sorted)

            items_sorted_by_ids_stream = iter(items_sorted_by_ids)

            self.stream_data_store.save_items_sorted_by_ids(items_sorted_by_ids_stream, ids_sorted_stream)

    def get_ids_sorted(self):
        with ExitStack() as stack:
            if hasattr(self.stream_data_store, '__enter__'):
                stack.enter_context(self.stream_data_store)
            ids_sorted = self.stream_data_store.get_ids_sorted()
            count_ = self.stream_data_store.get_count()
            ids_sorted_ndarray = aggregate_iterable(ids_sorted, count_)
            return ids_sorted_ndarray

    def is_stream_data_store(self):
        return False
