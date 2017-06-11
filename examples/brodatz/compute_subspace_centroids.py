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
from core.data_store.csv_datastore import CSVDataStore
from core.evaluation.ground_truth import DataStoreBasedGroundTruth
from core.transformer.mean_calculator import MeanCalculator
from core.transformer.array_to_pqindices import ArrayToPQIndices
from core.quantization.pq_quantizer import restore_from_clusters

if __name__ == '__main__':
    basedir = r"ds_data\glcm" + "\\"

    base_vectors_path = basedir + "glcm.sqlite"
    base_vectors_ds = SQLiteTableDataStore(base_vectors_path, 'imgid_glcm')

    quantizer_params = {'n_clusters': 64, 'n_quantizers': 2}

    clusters_path = basedir + r'glcm_pqclusters-{}-{}.sqlite'.format(quantizer_params['n_clusters'],
                                                                     quantizer_params['n_quantizers'])

    clusters_ds = SQLiteTableDataStore(clusters_path, 'clusterid_cluster')
    clusters_ndarray_ds = StreamNdarrayAdapterDataStore(clusters_ds, detect_final_shape_by_first_elem=True)
    subspaced_clusters = clusters_ndarray_ds.get_items_sorted_by_ids()
    pqquantizer = restore_from_clusters(subspaced_clusters)

    subspace_centroids_path = basedir + 'glcm_pqclusters-{}-{}_subspacedcentroids.sqlite'.format(
        quantizer_params['n_clusters'],
        quantizer_params['n_quantizers'])
    subspace_centroids_ds = SQLiteTableDataStore(subspace_centroids_path)

    t = ArrayToPQIndices(pqquantizer)
    steps.transform_step(base_vectors_ds, [t], subspace_centroids_ds, force=True)
