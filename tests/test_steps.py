import os
import unittest

import numpy as np
from data_store.sqlite_table_datastore import SQLiteTableDataStore, get_two_sqlite_data_stores
from data_store.numpy_datastore import NumpyDataStore

import steps
from data_store import numpy_datastore
from quantization import pq_quantizer
from search import exhaustive_searcher
from transformer import transformers as tr
from common.aggregate_iterable import aggregate_iterable

def doubled_transform(items):
    return (item * 2 for item in items)


class SQLTabelDataStoreComputeDescriptorsTestCase(unittest.TestCase):
    db_path = "temp_sqlite_db"
    table_name_in = "temp_table_in"
    table_name_out = "temp_table_out"
    items = np.arange(10 * 8).reshape((10, 8))

    @classmethod
    def setUpClass(cls):
        with SQLiteTableDataStore(SQLTabelDataStoreComputeDescriptorsTestCase.db_path,
                                  SQLTabelDataStoreComputeDescriptorsTestCase.table_name_in, "ndarray") as sqltable_ds:
            sqltable_ds.save_items_sorted_by_ids(SQLTabelDataStoreComputeDescriptorsTestCase.items)

    @classmethod
    def tearDownClass(cls):
        with SQLiteTableDataStore(SQLTabelDataStoreComputeDescriptorsTestCase.db_path,
                                  SQLTabelDataStoreComputeDescriptorsTestCase.table_name_in, "ndarray") as sqltable_ds:
            sqltable_ds.drop(True)
        with SQLiteTableDataStore(SQLTabelDataStoreComputeDescriptorsTestCase.db_path,
                                  SQLTabelDataStoreComputeDescriptorsTestCase.table_name_out, "ndarray") as sqltable_ds:
            sqltable_ds.drop(True)
        os.remove(SQLTabelDataStoreComputeDescriptorsTestCase.db_path)

    def test_setUpClass_tearDownClass(self):
        pass

    def test_doubled_transform(self):
        sqltable_in_ds, sqltable_out_ds = get_two_sqlite_data_stores(
            SQLTabelDataStoreComputeDescriptorsTestCase.db_path,
            SQLTabelDataStoreComputeDescriptorsTestCase.table_name_in,
            SQLTabelDataStoreComputeDescriptorsTestCase.table_name_out,
            "ndarray"
        )
        steps.transform_step(sqltable_in_ds, [doubled_transform, doubled_transform], sqltable_out_ds)

        truth_arr = SQLTabelDataStoreComputeDescriptorsTestCase.items * 4
        with sqltable_out_ds:
            items = sqltable_out_ds.get_items_sorted_by_ids()
            for arr1, arr2 in zip(items, truth_arr):
                self.assertTrue((arr1 == arr2).all())

    def test_translate_arrays_by_keys(self):
        ids = np.arange(10)
        sourceids = np.array([str(x * 2) + ".jpg" for x in range(10)])
        matrix_to_translate = np.array([
            [3, 2, 0],
            [1, 4, 2]
        ]
        )
        np_ds_in = numpy_datastore.NumpyDataStore(matrix_to_translate)
        np_ds_out = numpy_datastore.NumpyDataStore()
        transformers_ = [tr.TranslateByKeysTransformer(ids, sourceids)]
        steps.transform_step(np_ds_in, transformers_, np_ds_out)
        truth_array = np.array([
            ["6.jpg", "4.jpg", "0.jpg"],
            ["2.jpg", "8.jpg", "4.jpg"]
        ])
        translated_array = np_ds_out.get_items_sorted_by_ids()
        self.assertTrue((translated_array == truth_array).all())

    def test_samples(self):
        sqltable_in_ds, sqltable_out_ds = get_two_sqlite_data_stores(
            SQLTabelDataStoreComputeDescriptorsTestCase.db_path,
            SQLTabelDataStoreComputeDescriptorsTestCase.table_name_in,
            SQLTabelDataStoreComputeDescriptorsTestCase.table_name_out,
            "ndarray"
        )
        steps.sampling_step(sqltable_in_ds, 0.5, sqltable_out_ds)

        truth_size = len(SQLTabelDataStoreComputeDescriptorsTestCase.items) // 2
        with sqltable_out_ds:
            items_count = sqltable_out_ds.get_count()
            self.assertEqual(truth_size, items_count)

    def test_quantize_shape_only(self):
        sqltable_in_ds, sqltable_out_ds = get_two_sqlite_data_stores(
            SQLTabelDataStoreComputeDescriptorsTestCase.db_path,
            SQLTabelDataStoreComputeDescriptorsTestCase.table_name_in,
            SQLTabelDataStoreComputeDescriptorsTestCase.table_name_out,
            "ndarray"
        )

        quantizer = pq_quantizer.PQQuantizer(n_clusters=4, n_quantizers=2)
        steps.quantize_step(sqltable_in_ds, quantizer, sqltable_out_ds)

        with sqltable_out_ds:
            cluster_centers = sqltable_out_ds.get_items_sorted_by_ids()
            cluster_centers_ndarray = aggregate_iterable(cluster_centers, detect_final_shape_by_first_elem=True)
            truth_shape = (2, 4, 4)
            self.assertEquals(cluster_centers_ndarray.shape, truth_shape)

    def test_search_step(self):
        sqltable_in_ds, sqltable_out_ds = get_two_sqlite_data_stores(
            SQLTabelDataStoreComputeDescriptorsTestCase.db_path,
            SQLTabelDataStoreComputeDescriptorsTestCase.table_name_in,
            SQLTabelDataStoreComputeDescriptorsTestCase.table_name_out,
            "ndarray"
        )

        with sqltable_in_ds:
            items = sqltable_in_ds.get_items_sorted_by_ids()
            count_ = sqltable_in_ds.get_count()
            items = aggregate_iterable(items, count_)
            ids = np.arange(1, 11)
            searcher_ = exhaustive_searcher.ExhaustiveSearcher(items, ids)

            Q = np.arange(8).reshape((1, -1))
            np_ds = numpy_datastore.NumpyDataStore(Q)

            steps.search_step(np_ds, searcher_, 10, sqltable_out_ds)
            truth_nearest_ids = np.arange(10, 0, -1)
            with sqltable_out_ds:
                items = sqltable_out_ds.get_items_sorted_by_ids()
                first_item = next(items)
                (first_item == truth_nearest_ids).all()


if __name__ == '__main__':
    unittest.main()
