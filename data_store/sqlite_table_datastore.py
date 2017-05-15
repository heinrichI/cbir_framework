import collections
import io
import itertools
import sqlite3

import numpy as np

from data_store.datastore import DataStore


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def adapt_npint(npint):
    pythonint = int(npint)
    return pythonint


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def get_two_sqlite_data_stores(db_path, table1_name, table2_name, item_column_type="ndarray"):
    sqltable_ds1 = SQLiteTableDataStore(db_path=db_path, table_name=table1_name, item_column_type=item_column_type)
    sqltable_ds2 = SQLiteTableDataStore(db_path=db_path, table_name=table2_name, item_column_type=item_column_type)
    return sqltable_ds1, sqltable_ds2


class SQLiteTableDataStore(DataStore):
    def __init__(self, db_path, table_name, item_column_type="ndarray"):
        self.db_path = db_path
        self.table_name = table_name
        self.item_column_type = item_column_type
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_adapter(np.int32, adapt_npint)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("ndarray", convert_array)

    def connect(self):
        self.connection = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.connection.execute('PRAGMA journal_mode=WAL')

    def close(self):
        self.connection.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def get_ids_sorted(self) -> collections.Iterable:
        ids = self.connection.execute("SELECT item_id FROM {0} ORDER BY item_id".format(self.table_name))
        ids = map(lambda id_tup: id_tup[0], ids)
        return ids

    def get_items_sorted_by_ids(self, ids_sorted: collections.Iterable = None):
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
        self.connection.execute(
            r"CREATE TABLE IF NOT EXISTS {0} (item_id INTEGER PRIMARY KEY, item {1})".format(self.table_name,
                                                                                             self.item_column_type))
        self.connection.execute("DELETE FROM {0}".format(self.table_name))

        if not ids_sorted:
            ids_sorted = itertools.count(1)

        id_item_sorted_by_ids_stream = zip(ids_sorted, items_sorted_by_ids)
        self.connection.executemany(
            "INSERT OR REPLACE INTO {0} (item_id, item) VALUES (?,?)".format(self.table_name),
            id_item_sorted_by_ids_stream)

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
