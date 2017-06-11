import core.steps as steps
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore, get_as_array
from core.data_store.numpy_datastore import NumpyDataStore, to_numpy_datastore
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters

from core.data_store.vecs_file_stream_datastore import VecsFileStreamDatastore

if __name__ == '__main__':
    K_arr = [2 ** i for i in [4, 6, 8, 10, 11]]
    m_arr = [m for m in [1, 2, 4, 8, 16]]
    # K_arr = [2 ** i for i in [6]]
    # m_arr = [2 ** i for i in [0]]
    quantizer_params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in K_arr for m in m_arr]

    descriptors_ds = VecsFileStreamDatastore(r'C:\data\texmex\sift\sift\sift_learn.fvecs', n_components=128)
    # we need it as array many times later, so aggregate it once here
    descriptors_ds = to_numpy_datastore(descriptors_ds)

    centroids_base_path = 'ds_data\\learn_centroids_tolerance_manipulations\\'
    for quantizer_params in quantizer_params_arr:
        pqcentroids_filename = centroids_base_path + 'centroids_pq-{}-{}'.format(quantizer_params['n_clusters'],
                                                                                 quantizer_params['n_quantizers'])
        pqcentroids_ds = SQLiteTableDataStore(pqcentroids_filename)
        pqcentroids_info_ds = SQLiteTableDataStore(pqcentroids_filename, table_name='quantization_info',
                                                   item_column_type='TEXT')
        quantizer = PQQuantizer(**quantizer_params, verbose=True)
        steps.quantize_step(descriptors_ds, quantizer, pqcentroids_ds, print_ds_out_info='shape',
                            quantization_info_ds=pqcentroids_info_ds)
