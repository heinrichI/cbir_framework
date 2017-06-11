from os.path import join as pjoin
from core.quantization.pq_quantizer import build_pq_params_str
import os
from core import data_store as ds
import re

class PathHelper:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def centroids_path(self, descriptor_name, pq_params=None):
        return pjoin(self.base_dir, descriptor_name, build_pq_params_str(pq_params))

    def centroids_pathes(self, descriptor_name):
        return os.listdir(pjoin(self.base_dir, descriptor_name))


class DataStoreHelper:
    """
        helps to organize pathes, datastores
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def global_descriptors_ds(self, descriptor_name):
        path_ = pjoin(self.base_dir, 'global_descriptors', descriptor_name)
        ds_ = ds.SQLiteTableDataStore(path_)
        return ds_

    def local_descriptors_ds(self, descriptor_name, one_to_many=True):
        path_ = pjoin(self.base_dir, 'local_descriptors', descriptor_name) + "_imgid"
        if one_to_many:
            ds_ = ds.SQLiteTableOneToManyDataStore(path_, table_name='id_item_imgid')
        else:
            ds_ = ds.SQLiteTableDataStore(path_, table_name='id_item_imgid')
        return ds_

    def local_descriptors_sample_ds(self, descriptor_name, sample_part):
        path_ = pjoin(self.base_dir, 'local_descriptors', descriptor_name) + "_sample-" + str(sample_part)
        ds_ = ds.SQLiteTableDataStore(path_)
        return ds_

    def centroids_ds(self, descriptor_name, pq_params):
        path_ = pjoin(self.base_dir, 'centroids', descriptor_name, build_pq_params_str(pq_params))
        ds_ = ds.SQLiteTableDataStore(path_)
        return ds_

    def centroids_pairwise_distances_ds(self, descriptor_name, pq_params):
        path_ = pjoin(self.base_dir, 'centroids_pairwise_distances', descriptor_name, build_pq_params_str(pq_params))
        ds_ = ds.SQLiteTableDataStore(path_)
        return ds_

    def pqcodes_ds(self, descriptor_name, pq_params):
        path_ = pjoin(self.base_dir, 'pqcodes', descriptor_name, build_pq_params_str(pq_params))
        ds_ = ds.SQLiteTableDataStore(path_, ndarray_bytes_only=True)
        return ds_

    def pq_search_neighbors_ids_ds(self, dc_type, descriptor_name, pq_params):
        path_ = pjoin(self.base_dir, 'pq_search', dc_type, 'neighbors_ids', descriptor_name,
                      build_pq_params_str(pq_params))
        ds_ = ds.CSVDataStore(path_, ndarray_elem_type_read='int32')
        return ds_

    def pq_search_perfomances_ds(self, dc_type, descriptor_name, pq_params):
        path_ = pjoin(self.base_dir, 'pq_search', dc_type, 'perfomances', descriptor_name,
                      build_pq_params_str(pq_params))
        ds_ = ds.CSVDataStore(path_, ndarray_elem_type_read='float32')
        return ds_

    def pq_search_perfomances_ds_arr(self, search_type, descriptor_name):
        path_ = pjoin(self.base_dir, 'pq_search', search_type, 'perfomances', descriptor_name)
        ds_arr = [ds.CSVDataStore(filename, ndarray_elem_type_read='float32') for filename in os.listdir(path_)]
        return ds_arr

    def pq_search_perfomances_plot_path(self, search_type, descriptor_name, perfomance_type):
        path_ = pjoin(self.base_dir, 'pq_search', search_type, 'perfomance_plots', descriptor_name, perfomance_type)
        return path_

    def ex_search_neighbors_ids_ds(self, descriptor_name):
        path_ = pjoin(self.base_dir, 'ex_search', 'neighbors_ids', descriptor_name)
        ds_ = ds.CSVDataStore(path_, ndarray_elem_type_read='int32')
        return ds_

    def ex_search_perfomances_ds(self, descriptor_name):
        path_ = pjoin(self.base_dir, 'ex_search', 'perfomances', descriptor_name)
        ds_ = ds.CSVDataStore(path_, ndarray_elem_type_read='float32')
        return ds_

    # def ex_search_perfomances_ds_arr(self):
    #     path_ = pjoin(self.base_dir, 'ex_search', 'perfomances')
    #     ds_arr = [ds.CSVDataStore(filename, ndarray_elem_type_read='float32') for filename in os.listdir(path_)]
    #     return ds_arr

    def ex_search_perfomances_plot_path(self, perfomance_type):
        path_ = pjoin(self.base_dir, 'ex_search', 'perfomance_plots', perfomance_type)
        return path_

    def ex_search_perfomances_memory_plot_path(self, perfomance_type):
        path_ = pjoin(self.base_dir, 'ex_search', 'perfomance_memory_plots', perfomance_type)
        return path_

    def perfomance_arr(self, perfomances_ds):
        with perfomances_ds:
            item = perfomances_ds.get_items_sorted_by_ids([1])
            perfomance_arr = next(item).reshape((4, -1));
            return perfomance_arr

    def bovw_descriptors_ds(self, descriptor_name, pq_params):
        # descriptor_name: 'bovwbincounts' | 'bovwproductbincounts'
        path_ = pjoin(self.base_dir, 'global_descriptors', descriptor_name + '_' + build_pq_params_str(pq_params))
        ds_ = ds.SQLiteTableDataStore(path_)
        return ds_

    def bovw_descriptors_names(self, descriptor_name, pq_params_arr):
        # descriptor_name: 'bovwbincounts' | 'bovwproductbincounts'
        names = []
        for pq_params in pq_params_arr:
            name = descriptor_name + '_' + build_pq_params_str(pq_params)
            path_ = pjoin(self.base_dir, 'global_descriptors', name + '.sqlite')
            # print(path_)
            if os.path.isfile(path_):
                names.append(name)
        return names

    def extract_pq_params_from_descriptor_name(self, descriptor_name):
        result = re.search(r'.*pq-([0-9]+)-([0-9]+).*', descriptor_name)
        # print(result.groups())
        try:
            k = int(result.group(1))
            m = int(result.group(2))
            pq_params = {'n_clusters': k, 'n_quantizers': m}
            return pq_params
        except:
            return None