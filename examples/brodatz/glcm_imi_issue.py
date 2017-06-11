import import_hack
import core.steps as steps
import core.data_store
import core
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.data_store.sqlite_table_one_to_many_datastore import SQLiteTableOneToManyDataStore
from core.data_store.file_system_directory_datastore import FileSystemDirectoryDataStore
from core.data_store.numpy_datastore import NumpyDataStore
from core.data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters
from core.transformer.bytes_to_ndarray import BytesToNdarray
from core.transformer.ndarray_to_opencvmatrix import NdarrayToOpencvMatrix
from core.transformer.opencvmatrix_to_histogram import OpencvMatrixToHistogram
from core.transformer.opencvmatrix_to_glcm import OpencvMatrixToGLCM
from core.transformer.opencvmatrix_to_lbphistogram import OpencvMatrixToLBPHistogram
from core.search.exhaustive_searcher import ExhaustiveSearcher
from core.search.inverted_multi_index_searcher import InvertedMultiIndexSearcher
from core.common.ds_utils import print_ds_items_info
from core.evaluation.retrieval_perfomance import PrecisionRecallAveragePrecisionEvaluator
from core.evaluation.ground_truth import BrodatzGroundTruth
from core import transformer as trs
from core.transformer.mean_calculator import MeanCalculator
import numpy as np
import sys

if __name__ == '__main__':
    img_dir_path = r'C:\data\images\brodatz\data.brodatz\size_213x213'
    images_ds = FileSystemDirectoryDataStore(dir_path=img_dir_path)
    print("images count in '{0}': ".format(img_dir_path), images_ds.get_count())
    base_transformers = [BytesToNdarray(), NdarrayToOpencvMatrix()]

    glcm_transformers = base_transformers + [OpencvMatrixToGLCM(True)]
    glcm_ds = SQLiteTableDataStore("ds_data\glcm", "imgid_glcm")
    # steps.transform_step(images_ds, glcm_transformers, glcm_ds)
    print_ds_items_info(glcm_ds)

    glcmclusters_ds = SQLiteTableDataStore('ds_data\glcm_pq-clusters-8-50', 'clusterid_cluster')
    quantizer = PQQuantizer(n_clusters=64, n_quantizers=8)
    steps.quantize_step(glcm_ds, quantizer, glcmclusters_ds)
    print_ds_items_info(glcmclusters_ds)

    glcmclusters_ndarray_ds = StreamNdarrayAdapterDataStore(glcmclusters_ds, detect_final_shape_by_first_elem=True)
    glcmcluster_centers = glcmclusters_ndarray_ds.get_items_sorted_by_ids()
    glcm_ndarray_ds = StreamNdarrayAdapterDataStore(glcm_ds, detect_final_shape_by_first_elem=True)
    glcm_X = glcm_ndarray_ds.get_items_sorted_by_ids()
    glcm_X_ids = glcm_ndarray_ds.get_ids_sorted()
    glcm_imi_searcher = InvertedMultiIndexSearcher(glcm_X_ids, glcmcluster_centers, x=glcm_X)

    n_nearest = 10

    glcm_approximateneighborsids_ds = NumpyDataStore()
    print("SEARCH STEP")
    steps.search_step(glcm_ds, glcm_imi_searcher, n_nearest, glcm_approximateneighborsids_ds)
    print_ds_items_info(glcm_approximateneighborsids_ds, print_shape_only=False, first_items_to_print=5)
