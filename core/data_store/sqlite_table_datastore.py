import collections
import itertools
import sqlite3
import os.path
import json

import numpy as np
from core.data_store.datastore import DataStore

from core.common.sqlite_utils import adapt_array, adapt_npint, convert_array
from core.common.file_utils import make_if_not_exists
from core.common.sqlite_utils import NdarrayFromBytes, NdarrayToBytes
from core.common.itertools_utils import front


def get_two_sqlite_data_stores(db_path, table1_name, table2_name, item_column_type="ndarray"):
    sqltable_ds1 = SQLiteTableDataStore(db_path=db_path, table_name=table1_name, item_column_type=item_column_type)
    sqltable_ds2 = SQLiteTableDataStore(db_path=db_path, table_name=table2_name, item_column_type=item_column_type)
    return sqltable_ds1, sqltable_ds2


class SQLiteTableDataStore(DataStore):
    # table_names_count = 1

    def __init__(self, db_path, table_name="id_item", item_column_type="ndarray", add_extension=True,
                 ndarray_bytes_only=False):

        self.db_path = db_path
        self.table_name = table_name
        self.item_column_type = item_column_type
        self.metadata_table_name = table_name + 'metadata'
        self.ndarray_bytes_only = ndarray_bytes_only

        sqlite3.register_adapter(np.int32, adapt_npint)

        if add_extension and '.' not in self.db_path:
            self.db_path += '.sqlite'

    def connect(self):
        make_if_not_exists(self.db_path)
        self.connection = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES, cached_statements=10 ** 5)
        self.connection.execute('PRAGMA journal_mode=WAL')
        # self.connection.execute('PRAGMA journal_mode=OFF')
        # self.connection.execute('PRAGMA synchronous=OFF')
        # self.connection.execute('PRAGMA count_changes=OFF')
        # self.connection.execute('PRAGMA page_size={}'.format(2 ** 14))
        self.connection.execute(
            r"CREATE TABLE IF NOT EXISTS {0} (item_id INTEGER PRIMARY KEY, item {1})".format(self.table_name,
                                                                                             self.item_column_type))

    def close(self):
        self.connection.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def get_metadata(self):
        cur = self.connection.execute(
            r"SELECT item_metadata FROM {}".format(self.metadata_table_name))
        metadata_json = next(cur)[0]
        metadata = json.loads(metadata_json)
        return metadata

    def get_ids_sorted(self) -> collections.Iterable:
        ids = self.connection.execute("SELECT item_id FROM {0} ORDER BY item_id".format(self.table_name))
        ids = map(lambda id_tup: id_tup[0], ids)
        return ids

    def update_converter(self):
        if self.ndarray_bytes_only:
            metadata = self.get_metadata()
            sqlite3.register_converter("ndarray", NdarrayFromBytes(True, metadata['shape'],
                                                                   metadata['dtype']))
        elif self.item_column_type == 'ndarray':
            sqlite3.register_converter("ndarray", NdarrayFromBytes(False))

    def update_adapter(self):
        if self.ndarray_bytes_only:
            sqlite3.register_adapter(np.ndarray, NdarrayToBytes(bytes_only=True))
        elif self.item_column_type == 'ndarray':
            sqlite3.register_adapter(np.ndarray, NdarrayToBytes(bytes_only=False))

    def save_metadata(self, item):
        if self.ndarray_bytes_only:
            metadata = {'shape': item.shape, 'dtype': str(item.dtype)}
            metadata_json = json.dumps(metadata)
            self.connection.execute(
                r"CREATE TABLE IF NOT EXISTS {0} (id INTEGER PRIMARY KEY, item_metadata TEXT)".format(
                    self.metadata_table_name))
            self.connection.execute("DELETE FROM {0}".format(self.metadata_table_name))
            self.connection.execute(
                r"INSERT OR REPLACE INTO {0} VALUES (1,?)".format(self.metadata_table_name), (metadata_json,))
            self.connection.commit()

    def get_items_sorted_by_ids(self, ids_sorted: collections.Iterable = None):
        self.update_converter()

        if not ids_sorted:
            item_stream = self.connection.execute('SELECT item FROM {0} ORDER BY item_id'.format(self.table_name))
        else:
            ids = filter(None, ids_sorted)
            id_list = ids
            if not isinstance(collections, np.ndarray):
                id_list = list(ids)
            if not id_list:
                return []
            in_clause_values = self._generate_in_clause_values(id_list)
            item_stream = self.connection.execute(
                "SELECT item FROM {0} WHERE item_id IN {1} ORDER BY item_id".format(self.table_name, in_clause_values))

        # sqlite always returns tuples. Here would be tuple (item,)
        item_stream = map(lambda item_tup: item_tup[0], item_stream)
        return item_stream

    def save_items_sorted_by_ids(self, items_sorted_by_ids: collections.Iterable,
                                 ids_sorted: collections.Iterable = None):
        front_, items_sorted_by_ids = front(items_sorted_by_ids)
        self.save_metadata(front_)
        self.update_adapter()

        self.connection.execute(
            r"CREATE TABLE IF NOT EXISTS {0} (item_id INTEGER PRIMARY KEY, item {1})".format(self.table_name,
                                                                                             self.item_column_type))
        self.connection.execute("DELETE FROM {0}".format(self.table_name))
        self.connection.commit()

        if not ids_sorted:
            ids_sorted = itertools.count(1)

        id_item_sorted_by_ids_stream = zip(ids_sorted, items_sorted_by_ids)
        self.connection.executemany(
            "INSERT INTO {0} (item_id, item) VALUES (?,?)".format(self.table_name),
            id_item_sorted_by_ids_stream)
        # self.connection.execute('COMMIT;')
        # self.connection.execute('commit')
        self.connection.commit()

    def get_count(self):
        count_tup = self.connection.execute("SELECT COUNT(*) FROM {0}".format(self.table_name))
        count_ = next(count_tup)[0]
        return count_

    def drop(self, vacuum=False):
        self.connection.execute(r"DROP TABLE IF EXISTS {0}".format(self.table_name))
        if vacuum:
            self.vacuum()

    def vacuum(self):
        self.connection.execute(r"VACUUM")

    def _generate_in_clause_values(self, id_list: collections.Iterable) -> str:
        # TODO maybe replace WHERE item_id IN -> temp table+JOIN
        in_clause_values = "({0})".format(",".join(map(str, id_list)))
        return in_clause_values

    def is_stream_data_store(self):
        return True

        # def _get_stored_item_params(self):
        #     self.connection.execute(
        #         r"-- CREATE TABLE IF NOT EXISTS {0} (item_id INTEGER PRIMARY KEY, item {1})".format(self.table_name,
        #                                                                                          self.item_column_type))
        #
        # def _update_stored_item_params(self, ):

    def already_exists(self):
        exists = os.path.isfile(self.db_path)
        if exists:
            with self:
                count_tup = self.connection.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{}'".format(self.table_name))
                try:
                    count_ = next(count_tup)[0]
                    if count_ == 1:
                        return True
                except:
                    return False

        return False

