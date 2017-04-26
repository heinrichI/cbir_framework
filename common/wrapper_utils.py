import numpy as np
import collections

from common.aggregate_iterable import aggregate_iterable
from common import itertools_utils as iu


def adapt_data_store_to_recieve_iterables(func):
    def wrapper(*args):
        new_args = []
        for arg in args:
            new_arg = arg
            if not isinstance(arg, np.ndarray) and isinstance(arg, collections.Iterable):
                aggregated = aggregate_iterable(arg, )
                new_arg = aggregated

            new_args.append(new_arg)
        return func(*new_args)

    return wrapper
