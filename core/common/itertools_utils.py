import collections
from itertools import *
from core.common.data_wrap import DataWrap
import numpy as np


def chunk_stream(input_stream: collections.Iterable, chunk_size) -> collections.Iterable:
    """chunks stream"""
    args = [iter(input_stream)] * chunk_size
    return zip_longest(*args, fillvalue=None)


def flatten_stream(chunk_stream: collections.Iterable) -> collections.Iterable:
    """flattens chunks into one stream"""
    return chain.from_iterable(chunk_stream)


def chunkify_input(func, chunk_size):
    def wrapper(input_iterable: collections.Iterable) -> collections.Iterable:
        chunked_input = chunk_stream(input_iterable, chunk_size)
        chunked_output = map(func, chunked_input)
        flattened_output = flatten_stream(chunked_output)
        return flattened_output

    return wrapper


def map_transformer(item_transformer):
    """wrap function which operates on one item to make it works for many items"""

    def wrapper(items: collections.Iterable):
        return map(item_transformer, items)

    return wrapper


def pipe_line(input_stream: collections.Iterable, transformers=[], items_count=-1,
              chunk_size=10 ** 5) -> collections.Iterable:
    """
    each transformer(iterable) operates over all input_stream producing new stream
    :param input_stream:
    :param transformers: iterable consisting of functions or objects with 'transform' method
    :return: output_stream
    """
    output_stream = input_stream
    dw = DataWrap(output_stream, items_count=items_count)
    for transformer in transformers:
        if hasattr(transformer, 'transform_chunks_stream'):
            if not dw.is_stream_chunkified:
                dw.chunkify(chunk_size)
            transform_function = transformer.transform_chunks_stream
        else:
            if dw.is_stream_chunkified:
                dw.dechunkify()

            if hasattr(transformer, 'transform'):
                transform_function = transformer.transform
            else:
                transform_function = transformer

        dw.data_stream = transform_function(dw.data_stream)
        # output_stream_copy, output_stream = tee(output_stream)
        # pass_consumer(output_stream_copy)

    dw.dechunkify()
    return dw.data_stream


def pass_consumer(input_stream: collections.Iterable):
    i = 0
    for x in input_stream:
        i += 1
        last_x = x
    print("input_stream item type:", type(last_x))
    if (isinstance(last_x, tuple)):
        for y in last_x:
            print("tuples item type:", type(y))
            if (isinstance(y, np.ndarray)):
                print("     ndarrary shape:", y.shape)

    print("input_stream_length:", i)


def transform_consume_pipe_line(id_stream: collections.Iterable, input_stream: collections.Iterable,
                                transformers=[], consumer=pass_consumer) -> collections.Iterable:
    output_stream = pipe_line(input_stream, transformers)
    id_output_stream = zip(id_stream, output_stream)
    consumer(id_output_stream)


def front(stream: collections.Iterable):
    x = iter(stream)
    first = next(x)
    x_restored = chain([first], x)
    return first, x_restored
