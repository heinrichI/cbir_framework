class DataStore(object):
    def get_items_sorted_by_ids(self, ids_sorted=None):
        # if ids_sorted=None, return all items
        pass

    def get_ids_sorted(self):
        pass

    def save_items_sorted_by_ids(self, items_sorted_by_ids, ids_sorted=None):
        # if ids_sorted=None, ids_sorted=range(1,len(items_sorted_by_ids)+1)
        pass

    def get_count(self):
        pass

    def is_stream_data_store(self):
        pass