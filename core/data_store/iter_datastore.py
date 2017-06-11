from core.data_store.datastore import DataStore


class IterDatastore(DataStore):
    def __init__(self, iter_stream):
        self.iter_stream = iter_stream

    def already_exists(self):
        raise NotImplementedError

    def get_ids_sorted(self):
        raise NotImplementedError

    def save_items_sorted_by_ids(self, items_sorted_by_ids, ids_sorted=None):
        raise NotImplementedError

    def get_items_sorted_by_ids(self, ids_sorted=None):
        if ids_sorted is not None:
            raise NotImplementedError
        return self.iter_stream

    def get_count(self):
        raise NotImplementedError

    def is_stream_data_store(self):
        return True
