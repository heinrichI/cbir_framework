import numpy as np
import pandas as ps


def translate_matrix_by_keys(keys: np.ndarray, values: np.ndarray, matrix_to_translate, return_list_of_lists=False):
    """translate every row of matrix by keys"""
    series = ps.Series(values, keys)
    translated_matrix = np.empty(matrix_to_translate.shape, series.values.dtype)
    for i in range(len(matrix_to_translate)):
        translated_matrix[i] = series[matrix_to_translate[i]].values
    # print("tr", translated_matrix.dtype)
    if return_list_of_lists:
        list_of_lists = [translated_matrix[i].tolist() for i in range(len(translated_matrix))]
        return list_of_lists
    else:
        return translated_matrix
