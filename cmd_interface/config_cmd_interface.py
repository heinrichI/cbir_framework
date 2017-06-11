from cmd_interface.converter.eval_args_converter import EvalArgsConverter
from cmd_interface.converter.pq_args_converter import PQQuantizerArgsConverter

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

typename_argsconverter = {
    'numpy': EvalArgsConverter(),
    'reshaper': EvalArgsConverter(),
    'pq': PQQuantizerArgsConverter()
}

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
    'mean':MeanCalculator,

    'pq': PQQuantizer,
}
