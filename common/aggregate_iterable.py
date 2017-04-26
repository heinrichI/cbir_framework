import collections
from itertools import chain

import numpy as np


def aggregate_iterable(x: collections.Iterable, n_elements=None):
    """
        aggregates iterable to ndarray
        x: iterable of ndarrays or primitives
    """
    x = iter(x)
    first = next(x)
    x_restored = chain([first], x)
    if isinstance(first, np.ndarray):
        iterable_flat = chain.from_iterable(map(np.ravel, x_restored))
        if (n_elements):
            count_ = n_elements * first.size
            x_ndarray_flat = np.fromiter(iterable_flat, dtype=first.dtype, count=count_)
        else:
            x_ndarray_flat = np.fromiter(iterable_flat, dtype=first.dtype)
        x_ndarray = x_ndarray_flat.reshape((-1,) + first.shape)
        return x_ndarray
    else:
        try:
            first_type = type(first)
            if (n_elements):
                aggregated = np.fromiter(x_restored, dtype=first_type, count=n_elements)
            else:
                aggregated = np.fromiter(x_restored, dtype=first_type)
        except Exception:
            raise
        else:
            return aggregated