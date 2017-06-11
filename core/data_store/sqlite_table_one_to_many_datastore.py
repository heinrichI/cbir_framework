import collections
import itertools
import sqlite3
import os.path

from core.data_store import datastore as ds
import numpy as np
from core.common.sqlite_utils import adapt_array, adapt_npint, convert_array
from core.common.file_utils import make_if_not_exists

import core.common.aggregate_iterable as ai


def generate_elements_for_iterable(iterable_item: collections.Iterable, foreignid):
    return zip(iterable_item, itertools.repeat(foreignid))


def generate_tuple(id, item_foreignid):
    return (id, item_foreignid[0], item_foreignid[1])


def aggregate_items(foreignid_itemsiterable):
    itemsiterable = foreignid_itemsiterable[1]
    itemsiterable = map(lambda foreignid_item: foreignid_item[1], itemsiterable)
    aggregated_items = ai.aggregate_iterable(itemsiterable, detect_final_shape_by_first_elem=True)
    return (foreignid_itemsiterable[0], aggregated_items)


class SQLiteTableOneToManyDataStore(ds.DataStore):
    def __init__(self, db_path, table_name, item_column_type="ndarray", add_extension=True):
        self.db_path = db_path
        self.table_name = table_name
        self.item_column_type = item_column_type
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_adapter(np.int32, adapt_npint)
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("ndarray", convert_array)

        if add_extension and '.' not in self.db_path:
            self.db_path += '.sqlite'

    def connect(self):
        make_if_not_exists(self.db_path)
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
        """
        
        :return: foreignids
        """
        ids = self.connection.execute("SELECT DISTINCT foreign_id FROM {0} ORDER BY foreign_id".format(self.table_name))
        ids = map(lambda id_tup: id_tup[0], ids)
        return ids

    def get_items_sorted_by_ids(self, ids_sorted: collections.Iterable = None):
        """
        
        :param ids_sorted: foreignids 
        :return: items - ndarrays of items(many items for one foreignid)
        """
        if not ids_sorted:
            foreignid_item_stream = self.connection.execute(
                'SELECT foreign_id, item FROM {0} ORDER BY foreign_id'.format(self.table_name))
            foreignid_iterableitem_stream = itertools.groupby(foreignid_item_stream,
                                                              lambda foreignid_item: foreignid_item[0])
            foreignid_aggregateditems = map(aggregate_items, foreignid_iterableitem_stream)
        else:
            ids = filter(None, ids_sorted)
            id_list = ids
            if not isinstance(collections, np.ndarray):
                id_list = list(ids)
            if not id_list:
                return []
            in_clause_values = self._generate_in_clause_values(id_list)
            foreignid_item_stream = self.connection.execute(
                "SELECT foreign_id, item FROM {0} WHERE foreign_id IN {1} ORDER BY foreign_id".format(self.table_name,
                                                                                                      in_clause_values))
            foreignid_iterableitem_stream = itertools.groupby(foreignid_item_stream,
                                                              lambda foreignid_item: foreignid_item[0])
            foreignid_aggregateditems = map(aggregate_items, foreignid_iterableitem_stream)

        # sqlite always returns tuples. Here would be tuple (item,)
        item_stream = map(lambda item_tup: item_tup[1], foreignid_aggregateditems)
        return item_stream

    def save_items_sorted_by_ids(self, items_sorted_by_ids: collections.Iterable,
                                 ids_sorted: collections.Iterable = None):
        self.recreate_table()

        if not ids_sorted:
            ids_sorted = itertools.count(1)

        item_foreignid = map(generate_elements_for_iterable, items_sorted_by_ids, ids_sorted)
        # iterable of zips
        item_foreignid = itertools.chain.from_iterable(item_foreignid)
        # iterable of tuples (item, foreignid)

        # for i in item_foreignid:
        #     for j in i:
        #         print(j)
        id_item_foreignid_sorted_by_ids_stream = map(generate_tuple, itertools.count(1), item_foreignid)
        # for i in id_item_foreignid_sorted_by_ids_stream:
        #     for j in i:
        #         print(j)
        self.connection.executemany(
            "INSERT OR REPLACE INTO {0} (item_id, item, foreign_id) VALUES (?,?,?)".format(self.table_name),
            id_item_foreignid_sorted_by_ids_stream)

        self.connection.commit()

        self.recreate_foreignid_index()

    def get_count(self):
        count_tup = self.connection.execute("SELECT COUNT(DISTINCT foreign_id) FROM {0}".format(self.table_name))
        count_ = next(count_tup)[0]
        return count_

    def drop(self, vacuum=False):
        self.connection.execute(r"DROP TABLE IF EXISTS {0}".format(self.table_name))
        if vacuum:
            self.vacuum()

    def vacuum(self):
        self.connection.execute(r"VACUUM")

    def recreate_table(self):
        self.connection.execute(
            r"CREATE TABLE IF NOT EXISTS {0} (item_id INTEGER PRIMARY KEY, item {1}, foreign_id INTEGER)".format(
                self.table_name,
                self.item_column_type))
        self.connection.execute("DELETE FROM {0}".format(self.table_name))

    def recreate_foreignid_index(self):
        self.connection.execute(
            r"CREATE INDEX IF NOT EXISTS foreignid_index ON {0} (foreign_id)".format(self.table_name))

    def _generate_in_clause_values(self, id_list: collections.Iterable) -> str:
        # TODO maybe replace WHERE item_id IN -> temp table+JOIN
        in_clause_values = "({0})".format(",".join(map(str, id_list)))
        return in_clause_values

    def is_stream_data_store(self):
        return True

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
