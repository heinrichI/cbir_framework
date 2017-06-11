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
from core.transformer.vectors_to_pairwisedistances import VectorsToPairwiseDistances
from core.metric.symmetric_distance_computer import SymmetricDistanceComputer
from core.transformer.array_to_pqindices import ArrayToPQIndices

if __name__ == '__main__':
    quantizer_params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in [4, 8, 16, 32, 64, 128, 256] for m in
                            [1, 2, 4, 8, 16] if 4 * K ** m < 14 * 2 ** 30]

    quantizer_params = {'n_clusters': 128, 'n_quantizers': 2}
    filepath = 'ds_data\glcm\glcm_pqclusters-{}-{}.sqlite'.format(quantizer_params['n_clusters'],
                                                                  quantizer_params['n_quantizers'])
    glcmclusters_ds = SQLiteTableDataStore(filepath, 'clusterid_cluster')

    clusters_pairwise_distances_filepath = 'ds_data\glcm\glcm_pqclusters-{}-{}_pairwisedistances.sqlite'.format(
        quantizer_params['n_clusters'],
        quantizer_params['n_quantizers'])
    clusters_pairwise_distances_ds = SQLiteTableDataStore(clusters_pairwise_distances_filepath)

    steps.transform_step(glcmclusters_ds, [VectorsToPairwiseDistances()], clusters_pairwise_distances_ds)

    glcmclusters_ndarray_ds = StreamNdarrayAdapterDataStore(glcmclusters_ds, detect_final_shape_by_first_elem=True)
    glcmcluster_centers = glcmclusters_ndarray_ds.get_items_sorted_by_ids()
    pq_quantizer = restore_from_clusters(glcmcluster_centers)
    clusters_pairwise_distances = StreamNdarrayAdapterDataStore(clusters_pairwise_distances_ds,
                                                                detect_final_shape_by_first_elem=True).get_items_sorted_by_ids()
    metric = SymmetricDistanceComputer(pq_quantizer, clusters_pairwise_distances)

    glcm_ds = SQLiteTableDataStore("ds_data\glcm\glcm.sqlite", "imgid_glcm")
    glcm_codes_filepath = 'ds_data\glcm\glcm_pqclusters-{}-{}_codes.sqlite'.format(quantizer_params['n_clusters'],
                                                                                   quantizer_params['n_quantizers'])
    glcm_codes_ds = SQLiteTableDataStore(glcm_codes_filepath)
    steps.transform_step(glcm_ds, [ArrayToPQIndices(pq_quantizer)], glcm_codes_ds)
    print_ds_items_info(glcm_codes_ds)

    glcm_codes_ndarray_ds = StreamNdarrayAdapterDataStore(glcm_codes_ds, detect_final_shape_by_first_elem=True)
    glcm_pqcodes = glcm_codes_ndarray_ds.get_items_sorted_by_ids()
    ids = glcm_codes_ndarray_ds.get_ids_sorted()
    searcher_ = ExhaustiveSearcher(glcm_pqcodes, ids, metric=metric,
                                   preprocess_Q_func=pq_quantizer.predict_subspace_indices_T)

    n_nearest = 10
    search_results_filepath = 'ds_data\glcm\glcm_pqclusters-{}-{}_search-{}-sdc.csv'.format(
        quantizer_params['n_clusters'], quantizer_params['n_quantizers'], n_nearest)
    glcmcodes_sdc_neighborsids_ds = CSVDataStore(search_results_filepath, item_type_read='ndarray',
                                                 ndarray_elem_type_read='int32')
    steps.search_step(glcm_ds, searcher_, n_nearest, glcmcodes_sdc_neighborsids_ds, force=True)
