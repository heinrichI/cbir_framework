from core.data_store.datastore import DataStore


class ListDatastore(DataStore):
    def __init__(self, listable=[]):
        self.list_ = list(listable)

    def already_exists(self):
        return False

    def get_ids_sorted(self):
        raise range(1, len(self.list_))

    def save_items_sorted_by_ids(self, items_sorted_by_ids, ids_sorted=None):
        self.list_ = list(items_sorted_by_ids)

    def get_items_sorted_by_ids(self, ids_sorted=None):
        if ids_sorted is not None:
            raise NotImplementedError
        return self.list_

    def get_count(self):
        return len(self.list_)

    def is_stream_data_store(self):
        return True
