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
from core.search.inverted_multi_index_searcher import InvertedMultiIndexSearcher, calculate_possible_params
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
    quantizer_params_arr = calculate_possible_params(14 * 2 ** 30)

    img_dir_path = r'C:\data\images\brodatz\data.brodatz\size_213x213'
    images_ds = FileSystemDirectoryDataStore(dir_path=img_dir_path)
    print("images count in '{0}': ".format(img_dir_path), images_ds.get_count())
    glcm_transformers = [BytesToNdarray(), NdarrayToOpencvMatrix(), OpencvMatrixToGLCM(True)]

    base_path = 'ds_data\glcm_new\\'
    descriptors_filename = base_path + 'glcm'
    descriptors_ds = SQLiteTableDataStore(descriptors_filename)
    steps.transform_step(images_ds, glcm_transformers, descriptors_ds, print_ds_out_info='shape')

    for quantizer_params in quantizer_params_arr:
        common_prefix = descriptors_filename + '_pq-{}-{}'.format(quantizer_params['n_clusters'],
                                                                  quantizer_params['n_quantizers'])

        pqcentroids_filename = common_prefix + '_centroids'
        pqcentroids_ds = SQLiteTableDataStore(pqcentroids_filename)
        quantizer = PQQuantizer(**quantizer_params, verbose=False)
        steps.quantize_step(descriptors_ds, quantizer, pqcentroids_ds, print_ds_out_info='shape')

        pqcentroids = get_as_array(pqcentroids_ds)
        ids, descriptors = get_as_array(descriptors_ds, return_ids=True)
        imi_searcher = InvertedMultiIndexSearcher(ids, pqcentroids, x=descriptors)

        n_nearest = 100

        search_results_filename = common_prefix + '_search-{}'.format(n_nearest)
        approximateneighborsids_ds = CSVDataStore(search_results_filename, 'ndarray', 'int32')
        print("search_step:", search_results_filename)
        steps.search_step(descriptors_ds, imi_searcher, n_nearest, approximateneighborsids_ds,
                          print_ds_out_info='ndarray')

        common_prefix = search_results_filename
        search_mean_perfomance_filename = common_prefix + '_perfomance_mean'
        search_mean_perfomance_ds = CSVDataStore(search_mean_perfomance_filename, 'ndarray',
                                                 'float32')

        brodatz_gt_positives_ds = CSVDataStore(r'ds_data\brodatz_gt_positives.csv', 'ndarray', 'int32')
        ground_truth = DataStoreBasedGroundTruth(brodatz_gt_positives_ds)
        evaluator = PrecisionRecallAveragePrecisionEvaluator(ground_truth)
        steps.evaluation_step(approximateneighborsids_ds, evaluator, search_mean_perfomance_ds)

        label = 'glcm_pqclusters-{}-{}'.format(quantizer_params['n_clusters'], quantizer_params['n_quantizers'])

        xlabel = 'n_nearest'
        ylabel = 'precision'
        save_to_file = search_mean_perfomance_filename + '_{}.png'.format(ylabel)
        steps.plotting_step([search_mean_perfomance_ds], 0, 1, [label], xlabel, ylabel, save_to_file=save_to_file)

        ylabel = 'recall'
        save_to_file = search_mean_perfomance_filename + '_{}.png'.format(ylabel)
        steps.plotting_step([search_mean_perfomance_ds], 0, 2, [label], xlabel, ylabel, save_to_file=save_to_file)

        ylabel = 'mAP'
        save_to_file = search_mean_perfomance_filename + '_{}.png'.format(ylabel)
        steps.plotting_step([search_mean_perfomance_ds], 0, 3, [label], xlabel, ylabel, save_to_file=save_to_file)
