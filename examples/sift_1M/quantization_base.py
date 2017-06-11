import core.steps as steps
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.data_store.stream_ndarray_adapter_datastore import StreamNdarrayAdapterDataStore, get_as_array
from core.data_store.numpy_datastore import NumpyDataStore, to_numpy_datastore
from core.quantization.pq_quantizer import PQQuantizer, restore_from_clusters

from core.data_store.vecs_file_stream_datastore import VecsFileStreamDatastore

if __name__ == '__main__':
    K_arr = [2 ** i for i in [4, 6, 8, 10, 11]]
    m_arr = [m for m in [1, 2, 4, 8, 16]]
    quantizer_params_arr = [{'n_clusters': K, 'n_quantizers': m} for K in K_arr for m in m_arr]

    descriptors_ds = VecsFileStreamDatastore(r'C:\data\texmex\sift\sift\sift_base.fvecs', n_components=128)
    descriptors_ds = to_numpy_datastore(descriptors_ds)

    base_path = 'ds_data\\base_quantization\\'
    for pq_params in quantizer_params_arr:
        pq_params_str = 'pq-{}-{}'.format(pq_params['n_clusters'], pq_params['n_quantizers'])

        centroids_filepath = base_path + 'centroids\\' + pq_params_str + '-centroids'
        centroids_ds = SQLiteTableDataStore(centroids_filepath)
        centroids_info_ds = SQLiteTableDataStore(centroids_filepath, table_name='quantization_info',
                                                 item_column_type='TEXT')
        quantizer = PQQuantizer(**pq_params, verbose=True)
        steps.quantize_step(descriptors_ds, quantizer, centroids_ds, print_ds_out_info='shape',
                            quantization_info_ds=centroids_info_ds)
