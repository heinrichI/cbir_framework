import numpy as np
from skimage.feature import greycomatrix

from core.transformer.items_transformer import ItemsTransformer


def opencvmatrix_to_glcm(matrix: np.ndarray, normalize=True, levels=256):
    glcm = greycomatrix(matrix, [1], [0], normed=normalize, levels=levels)
    glcm = glcm.ravel()
    return glcm



class OpencvMatrixToGLCM(ItemsTransformer):
    def __init__(self, normalize=True, levels=256):
        self.normalize = normalize
        self.opencvmatrix_to_glcm = opencvmatrix_to_glcm
        self.levels = levels

    def transform_item(self, item):
        return self.opencvmatrix_to_glcm(item, self.normalize, levels=self.levels)