from client_server.server.search_config import SearchConfig
from core.data_store.file_system_directory_datastore import FileSystemDirectoryDataStore
from core.data_store.numpy_datastore import NumpyDataStore
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.data_store.stream_ndarray_adapter_datastore import get_as_array
from core.search.exhaustive_searcher import ExhaustiveSearcher
from core.transformer import *


class ConfigBuilder():
    def __init__(self):
        self.params = {}

    def base_images_dir_path(self, dir_path):
        base_images_ds = FileSystemDirectoryDataStore(dir_path,
                                                      ids_are_fullpathes=True)
        self.params['base_native_ids_ds'] = NumpyDataStore(base_images_ds.get_ids_sorted())

    def base_vectors_sql_ds(self, path_):
        base_vectors_ds = SQLiteTableDataStore(path_)
        base_ids, base_vectors = get_as_array(base_vectors_ds, return_ids=True)
        self.params['base_ids'] = base_ids
        self.params['base_vectors'] = base_vectors

    def transformers(self, trs_arr):
        self.params['transformers'] = trs_arr

    def build_exhaustive_searcher_cfg(self):
        searcher = ExhaustiveSearcher(self.params['base_vectors'], self.params['base_ids'])
        search_config = SearchConfig(self.params['transformers'], searcher, self.params['base_native_ids_ds'])
        return search_config


def cnfg_brodatz_histogram():
    base_images_ds = FileSystemDirectoryDataStore(r'C:\data\images\brodatz\data.brodatz\size_213x213',
                                                  ids_are_fullpathes=True)
    base_native_ids_ds = NumpyDataStore(base_images_ds.get_ids_sorted())

    base_vectors_ds = SQLiteTableDataStore(
        r'C:\Users\Dima\GoogleDisk\notebooks\cbir_framework\examples\brodatz\ds_data\imgid_histogram')
    base_ids, base_vectors = get_as_array(base_vectors_ds, return_ids=True)

    transformers = [FilepathToImageBytes(), BytesToNdarray(), NdarrayToOpencvMatrix(), OpencvMatrixToHistogram()]
    searcher = ExhaustiveSearcher(base_vectors, base_ids)
    base_native_ids_ds = base_native_ids_ds
    search_config = SearchConfig(transformers, searcher, base_native_ids_ds)
    return search_config


def cnfg_oxford_histogram():
    cfg_ = ConfigBuilder()
    cfg_.base_images_dir_path(r'C:\data\images\oxford\oxbuild_images')
    cfg_.base_vectors_sql_ds(
        r'C:\data\computation\oxford\histograms')
    cfg_.transformers([FilepathToImageBytes(), BytesToNdarray(), NdarrayToOpencvMatrix(), OpencvMatrixToHistogram()])
    search_config = cfg_.build_exhaustive_searcher_cfg()
    return search_config


def cnfg_oxford_lbphistogram():
    cfg_ = ConfigBuilder()
    cfg_.base_images_dir_path(r'C:\data\images\oxford\oxbuild_images')
    cfg_.base_vectors_sql_ds(
        r'C:\data\computation\oxford\lbphistograms')
    cfg_.transformers([FilepathToImageBytes(), BytesToNdarray(), NdarrayToOpencvMatrix(), OpencvMatrixToLBPHistogram()])
    search_config = cfg_.build_exhaustive_searcher_cfg()
    return search_config


def cnfg_oxford_glcm():
    cfg_ = ConfigBuilder()
    cfg_.base_images_dir_path(r'C:\data\images\oxford\oxbuild_images')
    cfg_.base_vectors_sql_ds(
        r'C:\data\computation\oxford\glcms')
    cfg_.transformers([FilepathToImageBytes(), BytesToNdarray(), NdarrayToOpencvMatrix(), OpencvMatrixToGLCM()])
    search_config = cfg_.build_exhaustive_searcher_cfg()
    return search_config


def build_config_dict():
    # cnfg=cnfg_brodatz_histogram()
    # cnfg = cnfg_oxford_histogram()
    # cnfg = cnfg_oxford_lbphistogram()
    cnfg = cnfg_oxford_glcm()

    return cnfg
