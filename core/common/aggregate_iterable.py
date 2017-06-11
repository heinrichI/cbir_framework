import collections
from itertools import chain

import numpy as np


def aggregate_iterable(x: collections.Iterable, detect_final_shape_by_first_elem=False, final_shape=None,
                       n_elements=None):
    """
        aggregates iterable to ndarray
        x: iterable of ndarrays or primitives
    """
    x = iter(x)
    first = next(x)
    x_restored = chain([first], x)
    if isinstance(first, np.ndarray):
        iterable_flat = chain.from_iterable(map(np.ravel, x_restored))
        if n_elements and n_elements != -1:
            count_ = int(n_elements * first.size)
            x_ndarray_flat = np.fromiter(iterable_flat, dtype=first.dtype, count=count_)
        else:
            x_ndarray_flat = np.fromiter(iterable_flat, dtype=first.dtype)
        x_ndarray = x_ndarray_flat
        if detect_final_shape_by_first_elem:
            x_ndarray = x_ndarray_flat.reshape((-1,) + first.shape)
        elif final_shape is not None:
            x_ndarray = x_ndarray_flat.reshape(final_shape)
        return x_ndarray
    else:
        # x - iterable of primitives
        try:
            first_type = type(first)
            if n_elements is not None:
                aggregated = np.fromiter(x_restored, dtype=first_type, count=n_elements)
            else:
                aggregated = np.fromiter(x_restored, dtype=first_type)
            if final_shape is not None:
                aggregated = aggregated.reshape(final_shape)
        except Exception:
            raise
        else:
            return aggregated
