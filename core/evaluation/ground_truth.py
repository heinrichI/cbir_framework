import numpy as np
from core.data_store.datastore import DataStore
from contextlib import ExitStack
from core.data_store.numpy_datastore import NumpyDataStore


class GroundTruth:
    def get_positive_ids(self, id):
        pass

    def get_neutral_ids(self, id):
        return np.array([])


class BrodatzGroundTruth(GroundTruth):
    def get_positive_ids(self, id):
        cls_number = (id - 1) // 9
        positive_ids = np.arange(cls_number * 9, cls_number * 9 + 10)
        return positive_ids


class DataStoreBasedGroundTruth(GroundTruth):
    def __init__(self, ds_positives: DataStore, ds_neutrals: DataStore = None, store_as_array=True):
        self.ds_positives = ds_positives
        self.ds_neutrals = ds_neutrals
        self.store_as_array = store_as_array
        if store_as_array:
            with ExitStack() as stack:
                if hasattr(self.ds_positives, '__enter__'):
                    stack.enter_context(self.ds_positives)

                self.ds_positives = NumpyDataStore(ds_positives.get_items_sorted_by_ids())
                if self.ds_neutrals is not None:
                    self.ds_neutrals = NumpyDataStore(ds_neutrals.get_items_sorted_by_ids())

    def get_positive_ids(self, id):
        with ExitStack() as stack:
            if hasattr(self.ds_positives, '__enter__'):
                stack.enter_context(self.ds_positives)
            if self.store_as_array:
                positive_ids = self.ds_positives.get_items_sorted_by_ids([id])[0]
            else:
                positive_ids = next(self.ds_positives.get_items_sorted_by_ids([id]))
            return positive_ids

    def get_neutral_ids(self, id):
        if self.ds_neutrals is not None:
            with ExitStack() as stack:
                if hasattr(self.ds_positives, '__enter__'):
                    stack.enter_context(self.ds_positives)

                if self.store_as_array:
                    neutral_ids = self.ds_neutrals.get_items_sorted_by_ids([id])
                else:
                    neutral_ids = list(self.ds_neutrals.get_items_sorted_by_ids([id]))

                return neutral_ids
        return []


from core.data_store.csv_datastore import CSVDataStore


def generate_brodatz_ground_truth(filepath='brodatz_gt_positives.csv'):
    ds = CSVDataStore(filepath)
    ids = range(1, 9 * 112 + 1)
    positive_ids = [np.arange(((id_ - 1) // 9) * 9 + 1, ((id_ - 1) // 9) * 9 + 10) for id_ in ids]
    with ds:
        ds.save_items_sorted_by_ids(positive_ids)


if __name__ == '__main__':
    generate_brodatz_ground_truth()
