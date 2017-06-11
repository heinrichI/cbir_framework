import numpy as np
import sklearn.preprocessing as preprocessing

from core.transformer.parametrized_items_transformer import ParametrizedItemsTransformer


def normalize_array(arr: np.ndarray, norm='l2'):
    arr_ = arr.reshape((1, -1))
    normalized_arr = preprocessing.normalize(arr_, norm=norm)
    normalized_arr = normalized_arr.reshape(arr.shape)
    return normalized_arr


class ArrayNormalizer(ParametrizedItemsTransformer):
    def __init__(self, norm):
        self.norm = norm
        super().__init__(normalize_array, norm)