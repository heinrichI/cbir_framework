import numpy as np

from client_server.json_socket.tserver import ThreadedServer
from client_server.server.config_builder import *
from core import steps
from core.common.ds_utils import print_ds_items_info
from core.transformer.keys_to_values import TranslateByKeysTransformer
from core.data_store.list_datastore import ListDatastore

"""
def build_query_items_ds_from_json_dict(obj):
    type_ = obj['type']
    params_ = obj['params']
    if type_ == 'FileSystemDirectoryDataStore':
        obj = FileSystemDirectoryDataStore(**params_)
    elif type_ == 'FileListDataStore':
        pass

    return obj
"""


class SearchServer(ThreadedServer):
    def __init__(self, search_config):
        super().__init__()
        self.search_config = search_config

    def start(self, ):
        super().start()

    def _process_message(self, obj):
        n_nearest = obj['n_nearest']
        """
        query_items_ds_dict = obj['query_items_ds']
        query_items_ds = build_query_items_ds_from_json_dict(query_items_ds_dict)
        """

        query_images_filepathes = obj['query_image_filepathes']
        query_image_filepathes_ds = ListDatastore(query_images_filepathes)
        nearest_native_ids = self._find_nearest_native_ids(query_image_filepathes_ds, n_nearest)
        # nearest_native_ids_list = nearest_native_ids.tolist()
        self.send_obj(nearest_native_ids)

    def _find_nearest_native_ids(self, query_items_ds, n_nearest) -> np.ndarray:
        query_vectors_ds = NumpyDataStore()
        steps.transform_step(query_items_ds, self.search_config.transformers, query_vectors_ds)

        neighbors_ids = NumpyDataStore()
        steps.search_step(query_vectors_ds, self.search_config.searcher, n_nearest, neighbors_ids)

        base_ids = list(self.search_config.base_native_ids_ds.get_ids_sorted())
        base_native_ids = list(self.search_config.base_native_ids_ds.get_items_sorted_by_ids())
        trs_ = [TranslateByKeysTransformer(base_ids, base_native_ids, return_list_of_lists=True)]
        neighbors_native_ids = ListDatastore()
        steps.transform_step(neighbors_ids, trs_, neighbors_native_ids)
        print_ds_items_info(neighbors_native_ids, print_shape_only=False)
        return neighbors_native_ids.get_items_sorted_by_ids()


if __name__ == '__main__':
    search_config = build_config_dict()
    if search_config is None:
        raise RuntimeError("Config is None")

    ss = SearchServer(search_config)
    ss.start()
