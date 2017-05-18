from contextlib import ExitStack

import numpy as np
from data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore

from search.searcher import Searcher
from common.itertools_utils import pipe_line
from data_store.datastore import DataStore
from quantization.quantizer import Quantizer


def transform_step(data_store_in: DataStore, transformers, data_store_out: DataStore) -> None:
    """

          data_store_in and data_store_out must be of same type: both ndarray or both stream.
          transformers must be of ds type
    """
    # TODO add dataframe abstraction and make any step work with any type of data(stream and ndarray)
    with ExitStack() as stack:
        if hasattr(data_store_in, '__enter__'):
            stack.enter_context(data_store_in)
        if hasattr(data_store_out, '__enter__'):
            stack.enter_context(data_store_out)
        items_in_sorted_by_ids = data_store_in.get_items_sorted_by_ids()
        items_out_sorted_by_ids = pipe_line(items_in_sorted_by_ids, transformers)
        data_store_out.save_items_sorted_by_ids(items_out_sorted_by_ids)


def sampling_step(data_store_in: DataStore, sample_part, data_store_out: DataStore) -> None:
    """
        data_store_in - stream or ndarray ds
        data_store_out - stream or ndarray ds
    """
    with ExitStack() as stack:
        if hasattr(data_store_in, '__enter__'):
            stack.enter_context(data_store_in)
        if hasattr(data_store_out, '__enter__'):
            stack.enter_context(data_store_out)
        count_ = data_store_in.get_count()
        sample_size = int(count_ * sample_part)

        ds_ndarray_in = StreamNdarrayAdapterDataStore(data_store_in, detect_final_shape_by_first_elem=True)
        ids_sorted_ndarray = ds_ndarray_in.get_ids_sorted()
        id_ndarray_sample = np.random.choice(ids_sorted_ndarray, sample_size, replace=False)
        id_ndarray_sample.sort()
        id_ndarray_sample_sorted = id_ndarray_sample

        sample_items_sorted_by_ids = ds_ndarray_in.get_items_sorted_by_ids(id_ndarray_sample_sorted)

        ds_ndarray_out = StreamNdarrayAdapterDataStore(data_store_out, detect_final_shape_by_first_elem=True)
        ds_ndarray_out.save_items_sorted_by_ids(sample_items_sorted_by_ids)


def quantize_step(data_store_in: DataStore, quantizer: Quantizer, data_store_out: DataStore) -> None:
    """
        data_store_in - stream or ndarray ds
        data_store_out - stream or ndarray ds
    """
    with ExitStack() as stack:
        if hasattr(data_store_in, '__enter__'):
            stack.enter_context(data_store_in)
        if hasattr(data_store_out, '__enter__'):
            stack.enter_context(data_store_out)

        ds_ndarray_in = StreamNdarrayAdapterDataStore(data_store_in, detect_final_shape_by_first_elem=True)
        items_ndarray = ds_ndarray_in.get_items_sorted_by_ids()
        quantizer.fit(items_ndarray)
        cluster_centers_ndarray = quantizer.get_cluster_centers()

        ds_ndarray_out = StreamNdarrayAdapterDataStore(data_store_out, detect_final_shape_by_first_elem=True)
        ds_ndarray_out.save_items_sorted_by_ids(cluster_centers_ndarray)


def search_step(data_store_in: DataStore, searcher_: Searcher, n_nearest, data_store_out: DataStore) -> None:
    """
        data_store_in - stream or ndarray ds
        data_store_out - stream or ndarray ds
    """
    with ExitStack() as stack:
        if hasattr(data_store_in, '__enter__'):
            stack.enter_context(data_store_in)
        if hasattr(data_store_out, '__enter__'):
            stack.enter_context(data_store_out)

        ds_ndarray_in = StreamNdarrayAdapterDataStore(data_store_in, detect_final_shape_by_first_elem=True)
        items_ndarray = ds_ndarray_in.get_items_sorted_by_ids()

        nearest_ids_ndarray = searcher_.find_nearest_ids(items_ndarray, n_nearest)
        ds_ndarray_out = StreamNdarrayAdapterDataStore(data_store_out)
        ds_ndarray_out.save_items_sorted_by_ids(nearest_ids_ndarray)
