import import_hack
import core.steps as steps
import core.data_store
import core
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.data_store.sqlite_table_one_to_many_datastore import SQLiteTableOneToManyDataStore
from core.data_store.file_system_directory_datastore import FileSystemDirectoryDataStore
from core.data_store.numpy_datastore import NumpyDataStore
from core.data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore, get_as_array
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

if __name__ == '__main__':
    quantizer_params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in [4, 8, 16, 32, 64, 128, 256] for m in
                            [1, 2, 4, 8, 16] if 4 * K ** m < 14 * 2 ** 30]

    img_dir_path = r'C:\data\images\brodatz\data.brodatz\size_213x213'
    images_ds = FileSystemDirectoryDataStore(dir_path=img_dir_path)
    print("images count in '{0}': ".format(img_dir_path), images_ds.get_count())
    base_transformers = [BytesToNdarray(), NdarrayToOpencvMatrix()]

    glcm_transformers = base_transformers + [OpencvMatrixToGLCM(True)]
    glcm_ds = SQLiteTableDataStore("ds_data\glcm\glcm.sqlite", "imgid_glcm")
    steps.transform_step(images_ds, glcm_transformers, glcm_ds)
    print_ds_items_info(glcm_ds)

    for quantizer_params in quantizer_params_arr:
        filepath = 'ds_data\glcm\glcm_pqclusters-{}-{}.sqlite'.format(quantizer_params['n_clusters'],
                                                                      quantizer_params['n_quantizers'])
        glcmclusters_ds = SQLiteTableDataStore(filepath, 'clusterid_cluster')
        quantizer = PQQuantizer(**quantizer_params, verbose=False)
        steps.quantize_step(glcm_ds, quantizer, glcmclusters_ds)
        print_ds_items_info(glcmclusters_ds)

        glcmcluster_centers = get_as_array(glcmclusters_ds)
        glcm_ids, glcm_vectors = get_as_array(glcm_ds, return_ids=True)
        glcm_imi_searcher = InvertedMultiIndexSearcher(glcm_ids, glcmcluster_centers, x=glcm_vectors)

        n_nearest = 30

        search_results_filepath = 'ds_data\glcm\glcm_pqclusters-{}-{}_search-{}.csv'.format(
            quantizer_params['n_clusters'], quantizer_params['n_quantizers'], n_nearest)
        glcm_approximateneighborsids_ds = CSVDataStore(search_results_filepath, item_type_read='ndarray',
                                                       ndarray_elem_type_read='int32')
        print("search_step:", search_results_filepath)
        steps.search_step(glcm_ds, glcm_imi_searcher, n_nearest, glcm_approximateneighborsids_ds)
        print_ds_items_info(glcm_approximateneighborsids_ds, print_shape_only=False, first_items_to_print=5)

        search_perfomance_filepath = 'ds_data\glcm\glcm_pqclusters-{}-{}_search-{}_perfomance.csv'.format(
            quantizer_params['n_clusters'], quantizer_params['n_quantizers'], n_nearest)
        search_perfomance_ds = CSVDataStore(search_perfomance_filepath, item_type_read='ndarray',
                                            ndarray_elem_type_read='float32')

        brodatz_gt_positives_ds = CSVDataStore(r'ds_data\brodatz_gt_positives.csv', 'ndarray', 'int32')
        ground_truth = DataStoreBasedGroundTruth(brodatz_gt_positives_ds)
        evaluator = PrecisionRecallAveragePrecisionEvaluator(ground_truth)
        steps.evaluation_step(glcm_approximateneighborsids_ds, evaluator, search_perfomance_ds)

        search_mean_perfomance_filepath = 'ds_data\glcm\glcm_pqclusters-{}-{}_search-{}_mean_perfomance.csv'.format(
            quantizer_params['n_clusters'], quantizer_params['n_quantizers'], n_nearest)
        search_mean_perfomance_ds = CSVDataStore(search_mean_perfomance_filepath,
                                                 item_type_read='ndarray',
                                                 ndarray_elem_type_read='float32')
        steps.transform_step(search_perfomance_ds, [MeanCalculator()], search_mean_perfomance_ds)

        label = 'glcm_pqclusters-{}-{}'.format(quantizer_params['n_clusters'], quantizer_params['n_quantizers'])

        xlabel = 'n_nearest'
        ylabel = 'precision'
        save_to_file = 'ds_data\glcm\glcm_pqclusters-{}-{}_search-{}_{}.png'.format(quantizer_params['n_clusters'],
                                                                                    quantizer_params['n_quantizers'],
                                                                                    n_nearest, ylabel)
        steps.plotting_step([search_mean_perfomance_ds], 0, 1, [label], xlabel, ylabel, save_to_file=save_to_file)

        ylabel = 'recall'
        save_to_file = 'ds_data\glcm\glcm_pqclusters-{}-{}_search-{}_{}.png'.format(quantizer_params['n_clusters'],
                                                                                    quantizer_params['n_quantizers'],
                                                                                    n_nearest, ylabel)
        steps.plotting_step([search_mean_perfomance_ds], 0, 2, [label], xlabel, ylabel, save_to_file=save_to_file)

        ylabel = 'mAP'
        save_to_file = 'ds_data\glcm\glcm_pqclusters-{}-{}_search-{}_{}.png'.format(quantizer_params['n_clusters'],
                                                                                    quantizer_params['n_quantizers'],
                                                                                    n_nearest, ylabel)
        steps.plotting_step([search_mean_perfomance_ds], 0, 3, [label], xlabel, ylabel, save_to_file=save_to_file)
