from client_server.server.search_config import SearchConfig
from core.data_store.file_system_directory_datastore import FileSystemDirectoryDataStore
from core.data_store.numpy_datastore import NumpyDataStore
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.data_store.stream_ndarray_adapter_datastore import get_as_array
from core.search.exhaustive_searcher import ExhaustiveSearcher
from core.transformer import *
from core.common.path_helper import DataStoreHelper
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters
from core import data_store as ds
from core import transformer as tr




def cnfg_brodatz_histogram():
    ds_helper = DataStoreHelper(r'C:\data\computation\brodatz')
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


def cnfg_oxford_productbovwbincount():
    ds_helper = DataStoreHelper(r'C:\data\computation\oxford_2')

    pq_params = ds_helper.build_pq_params_arr([128], [8])[0]

    base_images_ds = FileSystemDirectoryDataStore(r'C:\data\images\oxford\oxbuild_images',
                                                  ids_are_fullpathes=True)
    base_native_ids_ds = NumpyDataStore(base_images_ds.get_ids_sorted())

    base_vectors_ds = ds_helper.bovw_descriptors_ds('bovwproductbincounts', pq_params)
    base_ids, base_vectors = get_as_array(base_vectors_ds, return_ids=True)

    centroids_ds = ds_helper.centroids_ds('fixedsizesifts', pq_params)
    centroids = ds.get_as_array(centroids_ds)
    pq_quantizer = restore_from_clusters(centroids)

    transformers = [FilepathToImageBytes(), BytesToNdarray(), NdarrayToOpencvMatrix(),
                    OpencvMatrixToFixedSizeSiftsSet(300), tr.ArraysToProductBinCount(pq_quantizer)]
    searcher = ExhaustiveSearcher(base_vectors, base_ids)
    base_native_ids_ds = base_native_ids_ds
    search_config = SearchConfig(transformers, searcher, base_native_ids_ds)

    return search_config


def build_config_dict():
    # cnfg = cnfg_brodatz_histogram()
    cnfg = cnfg_oxford_productbovwbincount()

    return cnfg
