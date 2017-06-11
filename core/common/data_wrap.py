import collections
from core.common.aggregate_iterable import aggregate_iterable
import itertools


class DataWrap:
    def __init__(self, data_stream: collections.Iterable, is_stream_chunkified=False, items_count=-1):
        self.data_stream = data_stream
        self.is_stream_chunkified = is_stream_chunkified
        self.items_count = items_count

    def chunk_sizes_(self, chunk_size):
        if chunk_size > self.items_count:
            chunk_size = self.items_count
        chunk_sizes = []
        items_left = self.items_count
        chunks_count = int(self.items_count // chunk_size)
        while items_left > 0:
            if items_left < chunk_size:
                chunk_size = items_left
            chunk_sizes.append(chunk_size)
            items_left -= chunk_size
        return chunk_sizes

    def chunkify(self, chunk_size=-1):
        if not self.is_stream_chunkified:
            chunk_sizes = self.chunk_sizes_(chunk_size)

            it = iter(self.data_stream)

            chunks_stream = (aggregate_iterable(it, detect_final_shape_by_first_elem=True,
                                                n_elements=chunk_size) for chunk_size in chunk_sizes)
            self.data_stream = chunks_stream
            self.is_stream_chunkified = True

    def dechunkify(self):
        if self.is_stream_chunkified:
            self.data_stream = itertools.chain.from_iterable(self.data_stream)
            self.is_stream_chunkified = False
