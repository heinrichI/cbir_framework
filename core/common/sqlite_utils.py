import io
import numpy as np
import sqlite3


def array_to_bytes(arr: np.ndarray):
    return sqlite3.Binary(arr)
    # return np.getbuffer(arr.tobytes())


class NdarrayToBytes:
    def __init__(self, bytes_only=False):
        # self.bytes_only=bytes_only
        if bytes_only:
            self.adapt_method = self.to_bytes
        else:
            self.adapt_method = self.to_bytes_with_metadata

    def to_bytes(self, arr: np.ndarray):
        # return sqlite3.Binary(arr)
        return arr.tobytes()

    def to_bytes_with_metadata(self, arr: np.ndarray):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def __call__(self, arr: np.ndarray, *args, **kwargs):
        return self.adapt_method(arr)


class NdarrayFromBytes:
    def __init__(self, bytes_only=False, shape=None, dtype=None):
        if bytes_only:
            self.shape = shape
            self.dtype = dtype
            self.convert_method = self.from_bytes
        else:
            self.convert_method = self.from_bytes_with_metadata

    def from_bytes(self, bytes_):
        return np.frombuffer(bytes_, self.dtype).reshape(self.shape)

    def from_bytes_with_metadata(self, bytes_):
        out = io.BytesIO(bytes_)
        return np.load(out)

    def __call__(self, bytes_, *args, **kwargs):
        return self.convert_method(bytes_)


class NdarrayAdapter:
    def __init__(self, bytes_only):
        # self.bytes_only=bytes_only
        if bytes_only:
            self.adapt_method = self.to_bytes
        else:
            self.adapt_method = self.to_bytes_with_metadata

    def to_bytes(self, arr: np.ndarray):
        return sqlite3.Binary(arr)

    def to_bytes_with_metadata(self, arr: np.ndarray):
        out = io.BytesIO()
        np.save(out, arr)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def __call__(self, arr: np.ndarray, *args, **kwargs):
        return self.adapt_method(arr)


class ArrayFromBytes:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, binary_data, *args, **kwargs):
        arr = np.frombuffer(binary_data, self.dtype)
        return arr


def array_from_no_metadata_sqlitebinary(binary_data, dtype):
    arr = np.frombuffer(binary_data, dtype)
    return arr


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())
    # return np.getbuffer(arr)


def adapt_npint(npint):
    pythonint = int(npint)
    return pythonint


def convert_array(binary_data):
    out = io.BytesIO(binary_data)
    out.seek(0)
    return np.load(out)


class ArrayFromBinaryDataConverter:
    """
        using convert_array(saving array + array info) for storing arrays, we have extra 80 bytes for each array
        there is idea to store array info separately(for all arrays) but it is possible if only all arrays in datastore are same type,shape
    """

    def __init__(self, binary_data, dtype, flat_arr_len):
        self.binary_data = binary_data
        self.dtype = dtype
        self.flar_arr_len = flat_arr_len

    def __call__(self, binary_data):
        out = io.BytesIO(binary_data)
        out.seek(0)
        # return np.load(out)

        arr = np.frombuffer(out, dtype=self.dtype, count=self.flat_arr_len)
        return arr

