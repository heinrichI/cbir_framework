import numpy as np
import core.steps as steps
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.transformer.vectors_to_pairwisedistances import VectorsToPairwiseDistances
from core.transformer.array_to_pqindices import ArrayToPQIndices
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters
from core.data_store.stream_ndarray_adapter_datastore import get_as_array
from core.data_store.vecs_file_stream_datastore import VecsFileStreamDatastore
from core.common.ds_utils import print_ds_items_info
from core.metric.symmetric_distance_computer import SymmetricDistanceComputer
from core.metric.asymmetric_distance_computer import AsymmetricDistanceComputer
from core.search.exhaustive_searcher import ExhaustiveSearcher
from core.data_store.csv_datastore import CSVDataStore
from core.evaluation.retrieval_perfomance import PrecisionRecallAveragePrecisionEvaluator
from core.evaluation.ground_truth import DataStoreBasedGroundTruth
from core.data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore
from core.search.inverted_multi_index_searcher import InvertedMultiIndexSearcher
from math import log2

K_arr = [2 ** i for i in [4, 6, 8, 10, 11]]
m_arr = [m for m in [1, 2, 4, 8, 16]]
quantizer_params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in K_arr for m in m_arr]
bytes_free = 1 << 34
imi_quantizer_params_arr = []
quantizer_params_arr.remove({'n_clusters': 1024, 'n_quantizers': 8})
quantizer_params_arr.remove({'n_clusters': 1024, 'n_quantizers': 16})
quantizer_params_arr.remove({'n_clusters': 2048, 'n_quantizers': 1})
quantizer_params_arr.remove({'n_clusters': 2048, 'n_quantizers': 2})
quantizer_params_arr.remove({'n_clusters': 2048, 'n_quantizers': 4})
quantizer_params_arr.remove({'n_clusters': 2048, 'n_quantizers': 8})
quantizer_params_arr.remove({'n_clusters': 2048, 'n_quantizers': 16})

for pq_params in quantizer_params_arr:
    K = pq_params['n_clusters']
    m = pq_params['n_quantizers']
    if 4 * K ** m < bytes_free:
        imi_quantizer_params_arr.append(pq_params)


# base_path = 'ds_data\\learn_quantization\\'


base_path = 'ds_data\\base_quantization\\'


def quantization_main():
    if 'base' in base_path:
        descriptors_ds = VecsFileStreamDatastore(r'C:\data\texmex\sift\sift\sift_base.fvecs', n_components=128)
    else:
        descriptors_ds = VecsFileStreamDatastore(r'C:\data\texmex\sift\sift\sift_learn.fvecs', n_components=128)

    for pq_params in quantizer_params_arr:
        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        centroids_filepath = base_path + 'centroids\\' + pq_params_str + '_centroids'
        centroids_ds = SQLiteTableDataStore(centroids_filepath)
        centroids_info_ds = SQLiteTableDataStore(centroids_filepath, table_name='quantization_info',
                                                 item_column_type='TEXT')
        quantizer = PQQuantizer(**pq_params, verbose=True)
        steps.quantize_step(descriptors_ds, quantizer, centroids_ds, print_ds_out_info='shape',
                            quantization_info_ds=centroids_info_ds)


def centroids_pairwise_distances_main():
    for pq_params in quantizer_params_arr:
        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        centroids_filepath = base_path + 'centroids\\' + pq_params_str + '_centroids'
        centroids_ds = SQLiteTableDataStore(centroids_filepath)

        centroids_pairwise_distances_path = base_path + 'centroids_pairwise_distances\\' + pq_params_str + '_centroids_pairwise_distances'
        centroids_pairwise_distances_ds = SQLiteTableDataStore(centroids_pairwise_distances_path)

        steps.transform_step(centroids_ds, [VectorsToPairwiseDistances()], centroids_pairwise_distances_ds)


def codes_main():
    for pq_params in quantizer_params_arr:
        base_vectors_path = r'C:\data\texmex\sift\sift\sift_base.fvecs'
        base_vectors_ds = VecsFileStreamDatastore(base_vectors_path, 128)

        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])
        centroids_filepath = base_path + 'centroids\\' + pq_params_str + '_centroids'
        centroids_ds = SQLiteTableDataStore(centroids_filepath)
        centroids = get_as_array(centroids_ds)
        pq_quantizer = restore_from_clusters(centroids)
        transformers = [ArrayToPQIndices(pq_quantizer)]

        base_vectors_codes_path = base_path + 'base_vectors_codes\\' + pq_params_str + '_base_vectors_codes'
        base_vectors_codes_ds = SQLiteTableDataStore(base_vectors_codes_path, ndarray_bytes_only=True)

        steps.transform_step(base_vectors_ds, transformers, base_vectors_codes_ds)


def sdc_exhaustive_search_main():
    for pq_params in quantizer_params_arr:
        query_vectors_path = r'C:\data\texmex\sift\sift\sift_query.fvecs'
        query_vectors_ds = VecsFileStreamDatastore(query_vectors_path, 128)
        # query_vectors_ids, query_vectors = get_as_array(query_vectors_ds, return_ids=True)

        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        centroids_pairwise_distances_path = base_path + 'centroids_pairwise_distances\\' + pq_params_str + '_centroids_pairwise_distances'
        centroids_pairwise_distances_ds = SQLiteTableDataStore(centroids_pairwise_distances_path)
        centroids_pairwise_distances = get_as_array(centroids_pairwise_distances_ds)

        centroids_filepath = base_path + 'centroids\\' + pq_params_str + '_centroids'
        centroids_ds = SQLiteTableDataStore(centroids_filepath)
        centroids = get_as_array(centroids_ds)
        pq_quantizer = restore_from_clusters(centroids)
        metric = SymmetricDistanceComputer(pq_quantizer, centroids_pairwise_distances)

        base_vectors_codes_path = base_path + 'base_vectors_codes\\' + pq_params_str + '_base_vectors_codes'
        base_vectors_codes_ds = SQLiteTableDataStore(base_vectors_codes_path, ndarray_bytes_only=True)
        base_vectors_ids, base_vectors_codes = get_as_array(base_vectors_codes_ds, return_ids=True)

        searcher_ = ExhaustiveSearcher(base_vectors_codes, base_vectors_ids, metric=metric)
        n_nearest = 100

        neighbors_ids_filepath = base_path + 'sdc-neighbors-ids\\' + pq_params_str + '_sdc-neighbors-ids'
        neighbors_ids_ds = CSVDataStore(neighbors_ids_filepath)

        steps.search_step(query_vectors_ds, searcher_, n_nearest, neighbors_ids_ds, force=True)


def adc_exhaustive_search_main():
    for pq_params in quantizer_params_arr:
        query_vectors_path = r'C:\data\texmex\sift\sift\sift_query.fvecs'
        query_vectors_ds = VecsFileStreamDatastore(query_vectors_path, 128)

        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])
        centroids_filepath = base_path + 'centroids\\' + pq_params_str + '_centroids'
        centroids_ds = SQLiteTableDataStore(centroids_filepath)
        subspaced_centroids = get_as_array(centroids_ds).astype(dtype='float32', order='C')
        metric = AsymmetricDistanceComputer(subspaced_centroids)

        base_vectors_codes_path = base_path + 'base_vectors_codes\\' + pq_params_str + '_base_vectors_codes'
        base_vectors_codes_ds = SQLiteTableDataStore(base_vectors_codes_path, ndarray_bytes_only=True)
        base_vectors_ids, base_vectors_codes = get_as_array(base_vectors_codes_ds, return_ids=True)

        searcher_ = ExhaustiveSearcher(base_vectors_codes, base_vectors_ids, metric=metric)
        n_nearest = 100

        neighbors_ids_filepath = base_path + 'adc-neighbors-ids\\' + pq_params_str + '_adc-neighbors-ids'
        neighbors_ids_ds = CSVDataStore(neighbors_ids_filepath)

        steps.search_step(query_vectors_ds, searcher_, n_nearest, neighbors_ids_ds, force=True)


def imi_search():
    for pq_params in imi_quantizer_params_arr:
        query_vectors_path = r'C:\data\texmex\sift\sift\sift_query.fvecs'
        query_vectors_ds = VecsFileStreamDatastore(query_vectors_path, 128)

        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])
        centroids_filepath = base_path + 'centroids\\' + pq_params_str + '_centroids'
        centroids_ds = SQLiteTableDataStore(centroids_filepath)
        subspaced_centroids = get_as_array(centroids_ds).astype(dtype='float32', order='C')

        base_vectors_codes_path = base_path + 'base_vectors_codes\\' + pq_params_str + '_base_vectors_codes'
        base_vectors_codes_ds = SQLiteTableDataStore(base_vectors_codes_path, ndarray_bytes_only=True)
        base_vectors_ids, base_vectors_codes = get_as_array(base_vectors_codes_ds, return_ids=True)

        imi_searcher = InvertedMultiIndexSearcher(base_vectors_ids, subspaced_centroids,
                                                  x_pqcodes=base_vectors_codes)

        n_nearest = 100

        neighbors_ids_filepath = base_path + 'imi-neighbors-ids\\' + pq_params_str + '_imi-neighbors-ids'
        neighborsids_ds = CSVDataStore(neighbors_ids_filepath)
        print("SEARCH STEP")
        steps.search_step(query_vectors_ds, imi_searcher, n_nearest, neighborsids_ds)


def imi_evaluate_search_main():
    for pq_params in imi_quantizer_params_arr:
        gt_indices_path = r'C:\data\texmex\sift\sift\sift_my_groundtruth.ivecs'
        gt_indices_ds = VecsFileStreamDatastore(gt_indices_path, 100)

        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        sdc_neighbors_ids_filepath = base_path + 'imi-neighbors-ids\\' + pq_params_str + '_imi-neighbors-ids'
        sdc_neighbors_ids_ds = CSVDataStore(sdc_neighbors_ids_filepath, ndarray_elem_type_read='int32')
        sdc_neighbors_ids_ds = StreamNdarrayAdapterDataStore(sdc_neighbors_ids_ds,
                                                             detect_final_shape_by_first_elem=True,
                                                             slice_get=(slice(None), 0, None))

        sdc_perfomance_mean_filepath = base_path + 'imi-perfomance-mean2\\' + pq_params_str + '_imi-perfomance-mean'
        sdc_perfomance_mean_ds = CSVDataStore(sdc_perfomance_mean_filepath)

        ground_truth = DataStoreBasedGroundTruth(gt_indices_ds, store_as_array=True)
        evaluator = PrecisionRecallAveragePrecisionEvaluator(ground_truth)
        steps.evaluation_step(sdc_neighbors_ids_ds, evaluator, sdc_perfomance_mean_ds, force=True)

def imi_plot_mean_perfomances():
    label__x__y = {}
    m__length__recall = {}
    for pq_params in imi_quantizer_params_arr:
        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        sdc_perfomance_mean_filepath = base_path + 'imi-perfomance-mean2\\' + pq_params_str + '_imi-perfomance-mean'
        sdc_perfomance_mean_ds = CSVDataStore(sdc_perfomance_mean_filepath)
        with sdc_perfomance_mean_ds:
            item = sdc_perfomance_mean_ds.get_items_sorted_by_ids([1])
            item = next(item)
            item = item.ravel()
            y = item[1]

            label = pq_params['n_quantizers']
            x__y = label__x__y.setdefault(label, {})
            x = pq_params['n_clusters']
            x__y[x] = y

            m = pq_params['n_quantizers']
            k = pq_params['n_clusters']
            length__k = m__length__recall.setdefault(m, {})
            x = log2(k) * m
            length__k[x] = y

    ylabel = 'recall@100'
    plot_filepath = base_path + 'imi-perfomance-mean-plot2\\' + 'imi-perfomance-mean' + '_k-' + ylabel
    steps.plotting_step2(label__x__y, 'm = ', 'k', ylabel,
                         save_to_file=plot_filepath)

    plot_filepath = base_path + 'imi-perfomance-mean-plot2\\' + 'imi-perfomance-mean' + '_codelength-' + ylabel
    steps.plotting_step2(m__length__recall, 'm = ', 'code length (bits)', ylabel,
                         save_to_file=plot_filepath)



def exhaustive_search_main():
    query_vectors_path = r'C:\data\texmex\sift\sift\sift_query.fvecs'
    query_vectors_ds = VecsFileStreamDatastore(query_vectors_path, 128)
    # query_vectors_ids, query_vectors = get_as_array(query_vectors_ds, return_ids=True)

    base_vectors_path = r'C:\data\texmex\sift\sift\sift_base.fvecs'
    base_vectors_ds = VecsFileStreamDatastore(base_vectors_path, n_components=128)
    base_vectors_ids, base_vectors = get_as_array(base_vectors_ds, return_ids=True)

    searcher_ = ExhaustiveSearcher(base_vectors, base_vectors_ids, metric='l2')
    n_nearest = 100

    neighbors_ids_filepath = r'C:\data\texmex\sift\sift\sift_my_groundtruth.ivecs'
    neighbors_ids_ds = VecsFileStreamDatastore(neighbors_ids_filepath, n_components=n_nearest)

    steps.search_step(query_vectors_ds, searcher_, n_nearest, neighbors_ids_ds, force=True)


def sdc_evaluate_search_main():
    for pq_params in quantizer_params_arr:
        gt_indices_path = r'C:\data\texmex\sift\sift\sift_my_groundtruth.ivecs'
        gt_indices_ds = VecsFileStreamDatastore(gt_indices_path, 100)

        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        sdc_neighbors_ids_filepath = base_path + 'sdc-neighbors-ids\\' + pq_params_str + '_sdc-neighbors-ids'
        sdc_neighbors_ids_ds = CSVDataStore(sdc_neighbors_ids_filepath, ndarray_elem_type_read='int32')
        sdc_neighbors_ids_ds = StreamNdarrayAdapterDataStore(sdc_neighbors_ids_ds,
                                                             detect_final_shape_by_first_elem=True,
                                                             slice_get=(slice(None), 0, None))

        sdc_perfomance_mean_filepath = base_path + 'sdc-perfomance-mean2\\' + pq_params_str + '_sdc-perfomance-mean'
        sdc_perfomance_mean_ds = CSVDataStore(sdc_perfomance_mean_filepath)

        ground_truth = DataStoreBasedGroundTruth(gt_indices_ds, store_as_array=True)
        evaluator = PrecisionRecallAveragePrecisionEvaluator(ground_truth)
        steps.evaluation_step(sdc_neighbors_ids_ds, evaluator, sdc_perfomance_mean_ds, force=True)


def adc_evaluate_search_main():
    for pq_params in quantizer_params_arr:
        gt_indices_path = r'C:\data\texmex\sift\sift\sift_my_groundtruth.ivecs'
        gt_indices_ds = VecsFileStreamDatastore(gt_indices_path, 100)

        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        sdc_neighbors_ids_filepath = base_path + 'adc-neighbors-ids\\' + pq_params_str + '_adc-neighbors-ids'
        sdc_neighbors_ids_ds = CSVDataStore(sdc_neighbors_ids_filepath, ndarray_elem_type_read='int32')
        sdc_neighbors_ids_ds = StreamNdarrayAdapterDataStore(sdc_neighbors_ids_ds,
                                                             detect_final_shape_by_first_elem=True,
                                                             slice_get=(slice(None), 0, None))

        sdc_perfomance_mean_filepath = base_path + 'adc-perfomance-mean2\\' + pq_params_str + '_adc-perfomance-mean'
        sdc_perfomance_mean_ds = CSVDataStore(sdc_perfomance_mean_filepath)

        ground_truth = DataStoreBasedGroundTruth(gt_indices_ds, store_as_array=True)
        evaluator = PrecisionRecallAveragePrecisionEvaluator(ground_truth)
        steps.evaluation_step(sdc_neighbors_ids_ds, evaluator, sdc_perfomance_mean_ds, force=True)


def sdc_plot_mean_perfomances():
    label__x__y = {}
    m__length__recall = {}
    for pq_params in quantizer_params_arr:
        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        sdc_perfomance_mean_filepath = base_path + 'sdc-perfomance-mean2\\' + pq_params_str + '_sdc-perfomance-mean'
        sdc_perfomance_mean_ds = CSVDataStore(sdc_perfomance_mean_filepath)
        with sdc_perfomance_mean_ds:
            item = sdc_perfomance_mean_ds.get_items_sorted_by_ids([1])
            item = next(item)
            item = item.ravel()
            y = item[1]

            label = pq_params['n_quantizers']
            x__y = label__x__y.setdefault(label, {})
            x = pq_params['n_clusters']
            x__y[x] = y

            m = pq_params['n_quantizers']
            k = pq_params['n_clusters']
            length__k = m__length__recall.setdefault(m, {})
            x = log2(k) * m
            length__k[x] = y

    ylabel = 'recall@100'
    plot_filepath = base_path + 'sdc-perfomance-mean-plot2\\' + 'sdc-perfomance-mean' + '_k-' + ylabel
    steps.plotting_step2(label__x__y, 'm = ', 'k', ylabel,
                         save_to_file=plot_filepath)


def adc_plot_mean_perfomances():
    label__x__y = {}
    m__length__recall = {}
    for pq_params in quantizer_params_arr:
        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        sdc_perfomance_mean_filepath = base_path + 'adc-perfomance-mean2\\' + pq_params_str + '_adc-perfomance-mean'
        sdc_perfomance_mean_ds = CSVDataStore(sdc_perfomance_mean_filepath)
        with sdc_perfomance_mean_ds:
            item = sdc_perfomance_mean_ds.get_items_sorted_by_ids([1])
            item = next(item)
            item = item.ravel()
            y = item[1]

            label = pq_params['n_quantizers']
            x__y = label__x__y.setdefault(label, {})
            x = pq_params['n_clusters']
            x__y[x] = y

            m = pq_params['n_quantizers']
            k = pq_params['n_clusters']
            length__k = m__length__recall.setdefault(m, {})
            x = log2(k) * m
            length__k[x] = y

    ylabel = 'recall@100'
    plot_filepath = base_path + 'adc-perfomance-mean-plot2\\' + 'adc-perfomance-mean' + '_k-' + ylabel
    steps.plotting_step2(label__x__y, 'm = ', 'k', ylabel,
                         save_to_file=plot_filepath)

    plot_filepath = base_path + 'adc-perfomance-mean-plot2\\' + 'adc-perfomance-mean' + '_codelength-' + ylabel
    steps.plotting_step2(m__length__recall, 'm = ', 'code length (bits)', ylabel,
                         save_to_file=plot_filepath)


def test_ds_main():
    import random
    i = random.randint(0, 100000000000)
    path = base_path + 'test_bytes_only_{}.sqlite'.format(i)
    test_ds = SQLiteTableDataStore(path, ndarray_bytes_only=True)
    arr = np.arange(0, 10 ** 6).reshape((-1, 1))
    # it = list(arr)
    it = iter(arr)
    with test_ds:
        # test_ds.connection.execute('PRAGMA journal_mode=WAL')
        test_ds.save_items_sorted_by_ids(it)
    print_ds_items_info(test_ds)


def compare_gts_main():
    neighbors_ids_filepath = r'C:\data\texmex\sift\sift\sift_my_groundtruth.ivecs'
    neighbors_ids_ds = VecsFileStreamDatastore(neighbors_ids_filepath, n_components=100)

    gt_neighbors_ids_filepath = r'C:\data\texmex\sift\sift\sift_groundtruth.ivecs'
    gt_neighbors_ids_ds = VecsFileStreamDatastore(gt_neighbors_ids_filepath, n_components=100)

    print_ds_items_info(neighbors_ids_ds, print_shape_only=False, first_items_to_print=1)
    print_ds_items_info(gt_neighbors_ids_ds, print_shape_only=False, first_items_to_print=1)


if __name__ == '__main__':
    # quantization_main()
    # centroids_pairwise_distances_main()
    # codes_main()
    # sdc_exhaustive_search_main()
    # adc_exhaustive_search_main()
    #  adc_evaluate_search_main()
    # sdc_evaluate_search_main()
    # sdc_plot_mean_perfomances()
    # adc_plot_mean_perfomances()

    imi_search()
    imi_evaluate_search_main()
    imi_plot_mean_perfomances()

    # exhaustive_search_main()

    # compare_gts_main()
    # test_ds_main()
