from core.data_store.datastore import DataStore
from core.common.file_chunk_reader import read_in_chunks
import struct
import numpy as np
import itertools
import os
from core.common.itertools_utils import front


class VecsFileStreamDatastore(DataStore):
    """
        read-only version
        consider only fvecs and ivecs
    """

    def __init__(self, filepath, n_components):
        self.filepath = filepath
        self.n_components = n_components
        self.component_size = 4
        self.row_size = 4 + n_components * self.component_size
        if filepath.endswith('fvecs'):
            self.component_dtype = '<f4'
        elif filepath.endswith('ivecs'):
            self.component_dtype = '<i4'
        else:
            raise NotImplementedError('file name must ends in fvecs or ivecs')

    def __enter__(self):
        self.file = open(self.filepath, 'ba+')

    def __exit__(self, type, value, traceback):
        self.file.close()

    def get_items_sorted_by_ids(self, ids_sorted=None):
        if ids_sorted is not None:
            raise NotImplementedError

        self.file.seek(0)
        # unpacker = struct.Struct('i{}f'.format(self.n_components))
        for chunk in read_in_chunks(self.file, chunk_size=self.row_size * 1000):
            n_rows = len(chunk) // self.row_size
            for row_num in range(n_rows):
                row = chunk[row_num * self.row_size:(row_num + 1) * self.row_size:]
                typesize = np.frombuffer(row, dtype='<i4', count=1)[0]
                vec = np.frombuffer(row, dtype=self.component_dtype, count=self.n_components, offset=4)
                yield vec

    def is_stream_data_store(self):
        return True

    def already_exists(self):
        raise NotImplementedError

    def save_items_sorted_by_ids(self, items_sorted_by_ids, ids_sorted=None):
        if ids_sorted is not None:
            raise NotImplementedError

        self.file.seek(0)
        self.file.truncate()


        front_, items_sorted_by_ids = front(items_sorted_by_ids)
        # pack_fmt = '<i<{}i'.format(len(front_))
        # packer = struct.Struct(pack_fmt)
        for item in items_sorted_by_ids:
            component_size_arr = np.empty(len(item) + 1)
            component_size_arr[0] = 4
            component_size_arr[1:] = item
            little_endian_arr = component_size_arr.astype(dtype='<i4')
            self.file.write(little_endian_arr.tobytes())
            # packer.pack(4, )
            # typesize_bytes = struct.pack('<i4', 4)
            # arr_bytes = struct.pack(arr_pack_fmt, )

    def get_ids_sorted(self):
        n_vectors = self.get_count()
        return range(1, n_vectors + 1)

    def get_count(self):
        size = os.path.getsize(self.filepath)
        n_vectors = size // (4 + self.n_components * self.component_size)
        return n_vectors
