import collections

import cv2;
import numpy as np
from skimage.feature import greycomatrix
import pandas as ps
import common.numpy_utils as npu


# debug_print_wrap = debug_print.debug_print_wrap


def bytes_to_ndarray(image_bytes: bytes, dtype=np.uint8) -> np.ndarray:
    ndarr = np.fromstring(image_bytes, dtype)
    return ndarr


def ndarray_to_opencvmatrix(ndarray: np.ndarray) -> np.ndarray:
    img_np = cv2.imdecode(ndarray, cv2.IMREAD_GRAYSCALE)
    return img_np


def opencvmatrix_to_sifts(matrix: np.ndarray, sifts_extractor) -> np.ndarray:
    kp, descriptors = sifts_extractor.detectAndCompute(matrix, None)
    if descriptors is None:
        # one formal zero sift if no keypoints where found. For example: white blank page
        descriptors = np.zeros((1, 128), dtype=np.float32)

    # print("keypoints", len(kp))
    return descriptors


def opencvmatrix_to_histogram(matrix: np.ndarray, graycolor: bool, normalize=True) -> np.ndarray:
    if (graycolor):
        channels, histSize, ranges = [0], [256], [0, 256]
    else:
        channels, histSize, ranges = [0, 1, 2], [256, 256, 256], [0, 256, 0, 256, 0, 256]
    hist = cv2.calcHist([matrix], channels, None, histSize, ranges)
    if (normalize):
        cv2.normalize(hist, hist)
    return hist


def opencvmatrix_to_glcm(matrix: np.ndarray, normalize=True, levels=256):
    glcm = greycomatrix(matrix, [1], [0], normed=normalize, levels=levels)
    glcm = glcm.ravel()
    return glcm


def siftsset_to_bovwbincount(siftsset, quantizer, nclusters):
    indices_arr = quantizer(siftsset)

    # bincount = np.bincount(indices, minlength=nclusters)
    final_bincout = np.empty((len(indices_arr), nclusters))
    for i in range(len(indices_arr)):
        final_bincout[i] = np.bincount(indices_arr[i], minlength=nclusters)
    final_bincout = final_bincout.ravel()
    print(final_bincout.shape)
    return final_bincout


class ItemsTransformer():
    def transform(self, items: collections.Iterable):
        return map(self.transform_item, items)

    def transform_item(self):
        pass

    def get_result_item_info(self):
        pass


class TransformerWithParams:
    def getParamsInfo():
        pass


class BytesToNdarray(ItemsTransformer):
    def __init__(self):
        self.bytes_to_ndarray = bytes_to_ndarray

    def transform_item(self, item):
        return self.bytes_to_ndarray(item)


class NdarrayToOpencvMatrix(ItemsTransformer):
    def __init__(self):
        self.ndarray_to_opencvmatrix = ndarray_to_opencvmatrix

    def transform_item(self, item):
        return self.ndarray_to_opencvmatrix(item)


class OpencvMatrixToHistogram(ItemsTransformer):
    def __init__(self, graycolor: bool, normalize=True):
        self.graycolor = graycolor
        self.normalize = normalize

        self.opencvmatrix_to_histogram = opencvmatrix_to_histogram

    def transform_item(self, item):
        return self.opencvmatrix_to_histogram(item, self.graycolor,
                                              self.normalize)


class OpencvMatrixToGLCM(ItemsTransformer):
    def __init__(self, normalize=True, levels=256):
        self.normalize = normalize
        self.opencvmatrix_to_glcm = opencvmatrix_to_glcm
        self.levels = levels

    def transform_item(self, item):
        return self.opencvmatrix_to_glcm(item, self.normalize, levels=self.levels)


class OpencvMatrixToSiftsSet(ItemsTransformer, TransformerWithParams):
    def __init__(self, nfeatures=None, nOctaveLayers=None, contrastThreshold=None, edgeThreshold=None, sigma=None):
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma

        # some problems passing None args to SIFT_create
        kwargs = {'nfeatures': nfeatures, 'nOctaveLayers': nOctaveLayers, 'contrastThreshold': contrastThreshold,
                  'edgeThreshold': edgeThreshold, 'sigma': sigma}
        self.kwargs = dict((k, v) for k, v in kwargs.items() if v)
        self.sifts_extractor = cv2.xfeatures2d.SIFT_create(**self.kwargs)
        self.opencvmatrix_to_sifts = opencvmatrix_to_sifts

    def transform_item(self, item):
        return self.opencvmatrix_to_sifts(item, self.sifts_extractor)

    def getParamsInfo(self):
        return {
            'type': 'sifts_set',
            'params': self.kwargs
        }


class NdarrayToNdarray(ItemsTransformer):
    def __init__(self, result_shape):
        self.result_shape = result_shape

    def transform_item(self, item):
        return item.reshape(self.result_shape)


class SiftsSetToBovwBinCount(ItemsTransformer):
    def __init__(self, quantizer, nclusters):
        self.quantizer = quantizer
        self.nclusters = nclusters
        self.siftsset_to_bovwbincount = siftsset_to_bovwbincount

    def transform_item(self, item):
        return siftsset_to_bovwbincount(item, self.quantizer, self.nclusters)


class ParametrizedItemsTransformer(ItemsTransformer):
    def __init__(self, item_transform_func, *args, **kwargs):
        self.item_transform_func = item_transform_func
        self.args = args
        self.kwargs = kwargs

    def transform_item(self, item):
        return self.item_transform_func(item, *self.args, **self.kwargs)


class TranslateByKeysTransformer(ItemsTransformer):
    def __init__(self, keys: np.ndarray, values: np.ndarray):
        self.keys = keys
        self.values = values

    def transform(self, items: np.ndarray):
        return npu.translate_matrix_by_keys(self.keys, self.values, items)


def get_ids_array_sample(ids: np.ndarray, fraction: float):
    size = int(len(ids) * fraction)
    ids_sample = np.random.choice(ids, size=size, replace=False)
    return ids_sample
