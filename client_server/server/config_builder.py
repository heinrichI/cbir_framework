from client_server.server.search_config import SearchConfig
from core.data_store.file_system_directory_datastore import FileSystemDirectoryDataStore
from core.data_store.numpy_datastore import NumpyDataStore
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.data_store.stream_ndarray_adapter_datastore import get_as_array
from core.search.exhaustive_searcher import ExhaustiveSearcher
from core.transformer import *
from core.common.path_helper import DataStoreHelper
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters

ds_helper = DataStoreHelper(r'C:\data\computation\brodatz')

def cnfg_brodatz_histogram():
    base_images_ds = FileSystemDirectoryDataStore(r'C:\data\images\brodatz\data.brodatz\size_213x213',
                                                  ids_are_fullpathes=True)
    base_native_ids_ds = NumpyDataStore(base_images_ds.get_ids_sorted())

    base_vectors_ds = ds_helper.global_descriptors_ds('histograms')
    base_ids, base_vectors = get_as_array(base_vectors_ds, return_ids=True)

    transformers = [FilepathToImageBytes(), BytesToNdarray(), NdarrayToOpencvMatrix(), OpencvMatrixToHistogram()]
    searcher = ExhaustiveSearcher(base_vectors, base_ids)
    base_native_ids_ds = base_native_ids_ds
    search_config = SearchConfig(transformers, searcher, base_native_ids_ds)
    return search_config


def build_config_dict():
    cnfg=cnfg_brodatz_histogram()

    return cnfg
