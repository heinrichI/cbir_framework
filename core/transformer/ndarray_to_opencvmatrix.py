import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import cv2
import numpy as np

from core.transformer.items_transformer import ItemsTransformer


def ndarray_to_opencvmatrix(ndarray: np.ndarray) -> np.ndarray:
    img_np = cv2.imdecode(ndarray, cv2.IMREAD_GRAYSCALE)
    return img_np


class NdarrayToOpencvMatrix(ItemsTransformer):
    def __init__(self):
        self.ndarray_to_opencvmatrix = ndarray_to_opencvmatrix

    def transform_item(self, item):
        return self.ndarray_to_opencvmatrix(item)