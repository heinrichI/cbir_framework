from core.data_store.numpy_datastore import NumpyDataStore
import numpy as np


class MeanCalculator:
    def transform(self, items_stream):
        ds = NumpyDataStore(items_stream)
        items_ndarray = ds.get_items_sorted_by_ids()
        average_items = np.mean(items_ndarray, axis=0)
        return [average_items]
