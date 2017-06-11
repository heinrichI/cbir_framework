import json
from json.decoder import WHITESPACE

from core.data_store.file_system_directory_datastore import FileSystemDirectoryDataStore
from core.data_store.numpy_datastore import NumpyDataStore
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.data_store.csv_datastore import CSVDataStore
from core.quantization.pq_quantizer import PQQuantizer
from core.transformer.array_to_reshapedarray import ArrayToReshapedArray
from core.transformer.bytes_to_ndarray import BytesToNdarray
from core.transformer.ndarray_to_opencvmatrix import NdarrayToOpencvMatrix
from core.transformer.opencvmatrix_to_glcm import OpencvMatrixToGLCM
from core.transformer.opencvmatrix_to_histogram import OpencvMatrixToHistogram
from core.transformer.opencvmatrix_to_siftset import OpencvMatrixToSiftsSet
from core.transformer.mean_calculator import MeanCalculator

typename_type = {
    'sqlite': SQLiteTableDataStore,
    'numpy': NumpyDataStore,
    'filedirectory': FileSystemDirectoryDataStore,
    'csv': CSVDataStore,

    'reshaper': ArrayToReshapedArray,
    'bytes_to_ndarray': BytesToNdarray,
    'ndarray_to_opencvmatrix': NdarrayToOpencvMatrix,
    'opencvmatrix_to_histogram': OpencvMatrixToHistogram,
    'opencvmatrix_to_glcm': OpencvMatrixToGLCM,
    'opencvmatrix_to_siftset': OpencvMatrixToSiftsSet,
    'mean': MeanCalculator,

    'pq': PQQuantizer,
}


def create_single_by_type_name_and_kwargs(typename, kwargs_dict):
    type_ = typename_type[typename]
    # print("typename", typename)
    # print("**kwargs_dict", **kwargs_dict)
    obj = type_(**kwargs_dict)
    return obj


class ObjEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, 'to_json_dict'):
            json_dict = o.to_json_dict()
            return json_dict
        return super().default(o)


class ObjDecoder(json.JSONDecoder):
    # def __init__(self):
    #     self.context = {}

    def decode(self, s, _w=WHITESPACE.match):
        obj = super().decode(s, _w)
        if isinstance(obj, dict) and 'type' in obj.keys():
            type_ = obj['type']
            kwargs = obj['kwargs']
            obj = create_single_by_type_name_and_kwargs(type_, kwargs)

        return obj


if __name__ == '__main__':
    fsds = FileSystemDirectoryDataStore('some_path', recursive=False)
    fsds_json = ObjEncoder().encode(fsds)
    print(fsds_json)
    fsds_restored = ObjDecoder().decode(fsds_json)
    print(fsds_restored)
