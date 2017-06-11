import numpy as np
import skimage
from skimage.feature import local_binary_pattern

from core.transformer.items_transformer import ItemsTransformer


def opencvmatrix_to_lbphistogram(matrix: np.ndarray, n_points, radius, method):
    lbp = local_binary_pattern(matrix, n_points, radius, method)
    hist = np.array(skimage.exposure.histogram(lbp))
    hist=hist.ravel()
    return hist




class OpencvMatrixToLBPHistogram(ItemsTransformer):
    def __init__(self, n_points=8, radius=1, method='uniform'):
        self.n_points = n_points
        self.radius = radius
        self.method = method
        self.opencvmatrix_to_lbphistogram = opencvmatrix_to_lbphistogram

    def transform_item(self, item):
        return self.opencvmatrix_to_lbphistogram(item, self.n_points, self.radius, self.method)