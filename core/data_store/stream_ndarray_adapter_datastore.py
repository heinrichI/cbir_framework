from contextlib import ExitStack

import numpy as np
from core.data_store import datastore

from core.common.aggregate_iterable import aggregate_iterable


class StreamNdarrayAdapterDataStore(datastore.DataStore):
    """
        adapts data_store that works with streams to work with ndarrays
    """

    def __init__(self, stream_data_store: datastore.DataStore, detect_final_shape_by_first_elem=False, shape_get=None,
                 slice_get=None,
                 element_n_dims_save=None):
        self.stream_data_store = stream_data_store
        # if not stream_data_store.is_stream_data_store():
        #     self.get_count = stream_data_store.get_count
        #     self.get_ids_sorted = stream_data_store.get_ids_sorted
        #     self.get_items_sorted_by_ids = stream_data_store.get_items_sorted_by_ids
        #     self.save_items_sorted_by_ids = stream_data_store.save_items_sorted_by_ids
        # else:
        self.stream_data_store = stream_data_store
        self.detect_final_shape_by_first_elem = detect_final_shape_by_first_elem
        self.shape = shape_get
        self.element_n_dims_save = element_n_dims_save
        self.slice_get = slice_get

    def get_count(self):
        with ExitStack() as stack:
            if hasattr(self.stream_data_store, '__enter__'):
                stack.enter_context(self.stream_data_store)
            return self.stream_data_store.get_count()

    def get_items_sorted_by_ids(self, ids_sorted: np.ndarray = None):
        with ExitStack() as stack:
            if hasattr(self.stream_data_store, '__enter__'):
                stack.enter_context(self.stream_data_store)

            if not self.stream_data_store.is_stream_data_store():
                items_sorted_by_ids = self.stream_data_store.get_items_sorted_by_ids(ids_sorted)
            else:
                ids_sorted_stream = None
                if ids_sorted is not None:
                    count_ = ids_sorted.shape[0]
                    ids_sorted = ids_sorted.ravel()
                    ids_sorted_stream = iter(ids_sorted)
                else:
                    count_ = self.stream_data_store.get_count()

                items_sorted_by_ids_stream = self.stream_data_store.get_items_sorted_by_ids(ids_sorted_stream)
                items_sorted_by_ids_ndarray = aggregate_iterable(items_sorted_by_ids_stream,
                                                                 detect_final_shape_by_first_elem=self.detect_final_shape_by_first_elem,
                                                                 final_shape=self.shape,
                                                                 n_elements=count_)
                items_sorted_by_ids = items_sorted_by_ids_ndarray

            if self.shape is not None:
                items_sorted_by_ids = items_sorted_by_ids.reshape(self.shape)
            if self.slice_get is not None:
                items_sorted_by_ids = items_sorted_by_ids[self.slice_get]

            return items_sorted_by_ids

    def save_items_sorted_by_ids(self, items_sorted_by_ids: np.ndarray, ids_sorted: np.ndarray = None):
        with ExitStack() as stack:
            if hasattr(self.stream_data_store, '__enter__'):
                stack.enter_context(self.stream_data_store)

            if self.element_n_dims_save is not None:
                new_shape = self.select_shape_(items_sorted_by_ids.shape, self.element_n_dims_save)
                items_sorted_by_ids = items_sorted_by_ids.reshape(new_shape)

            if not self.stream_data_store.is_stream_data_store():
                self.stream_data_store.save_items_sorted_by_ids(items_sorted_by_ids, ids_sorted)
            else:
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

            if not self.stream_data_store.is_stream_data_store():
                ids_sorted_ndarray = self.stream_data_store.get_ids_sorted()
            else:
                ids_sorted = self.stream_data_store.get_ids_sorted()
                count_ = self.stream_data_store.get_count()
                ids_sorted_ndarray = aggregate_iterable(ids_sorted,
                                                        n_elements=count_)
            return ids_sorted_ndarray

    def is_stream_data_store(self):
        return False

    def select_shape_(self, shape, element_n_dims_save):
        n_dims = len(shape)
        n_dims_to_flat = n_dims - element_n_dims_save
        dims_to_flat = shape[:n_dims_to_flat]
        first_dim = dims_to_flat[0]
        for dim in dims_to_flat[1:]:
            first_dim *= dim
        new_shape = (first_dim,) + shape[n_dims_to_flat:]
        return new_shape


def get_as_array(ds: datastore.DataStore, return_ids=False):
    nd_array_ds = StreamNdarrayAdapterDataStore(ds, detect_final_shape_by_first_elem=True)
    items = nd_array_ds.get_items_sorted_by_ids()
    if not return_ids:
        return items
    else:
        ids = nd_array_ds.get_ids_sorted()
        return (ids, items)
