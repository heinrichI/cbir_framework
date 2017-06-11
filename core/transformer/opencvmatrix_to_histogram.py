import cv2
import numpy as np

from core.transformer.items_transformer import ItemsTransformer


def opencvmatrix_to_histogram(matrix: np.ndarray, graycolor: bool = True, normalize=True) -> np.ndarray:
    if (graycolor):
        channels, histSize, ranges = [0], [256], [0, 256]
    else:
        channels, histSize, ranges = [0, 1, 2], [256, 256, 256], [0, 256, 0, 256, 0, 256]
    hist = cv2.calcHist([matrix], channels, None, histSize, ranges)
    if (normalize):
        cv2.normalize(hist, hist)
    return hist


class OpencvMatrixToHistogram(ItemsTransformer):
    def __init__(self, graycolor: bool = True, normalize=True):
        self.graycolor = graycolor
        self.normalize = normalize

        self.opencvmatrix_to_histogram = opencvmatrix_to_histogram

    def transform_item(self, item):
        return self.opencvmatrix_to_histogram(item, self.graycolor,
                                              self.normalize)
