import numpy as np
import pandas as ps

def translate_matrix_by_keys(keys: np.ndarray, values: np.ndarray, matrix_to_translate):
    """translate every row of matrix by keys"""
    series = ps.Series(values, keys)
    # print("vd", values.dtype)
    # print("sd", series.dtype)
    # print("svd",series.values.dtype)
    translated_matrix = np.empty(matrix_to_translate.shape, series.values.dtype)
    for i in range(len(matrix_to_translate)):
        translated_matrix[i] = series[matrix_to_translate[i]].values
    # print("tr", translated_matrix.dtype)
    return translated_matrix


