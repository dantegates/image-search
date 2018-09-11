import collections
import statistics

import numba
import numpy as np
import tqdm


DEBUG = False


# see,
# https://en.wikipedia.org/wiki/Hamming_distance#Algorithm_example
@numba.vectorize
def _hamming_distance(n1, n2):
    # this number is made of each bit in either n1 or n2
    # but not both
    v = n1 ^ n2
    d = 0
    while v != 0:
        # subtracting 1 clears the least bit, a, in v and sets all bits
        # before a which are cleared by the logical &
        # 2^n = sum(2^m for 0 <= m <= n-1)
        d += 1
        v &= v - 1
    return d


class LookupTable:
    def __init__(self, encoder, items, threshold=3):
        self.encoder = encoder
        self.threshold = threshold
        # lookup table for an item (e.g. image) by its ID
        self._items = {}
        # lookup table for list of image IDs by encoding
        self._semantic_hash_table = collections.defaultdict(list)
        # numpy array that we can quickly compute hamming distance over
        self._table_keys = np.array(self._semantic_hash_table.keys())
        self._populate_data_structures(items)
        # defaultdict is convenient when initializing the DB instance but
        # dangerous to keep around.
        self._semantic_hash_table.default_factory = None  

    def describe(self):
        number_of_buckets = len(self._semantic_hash_table.keys())
        bucket_sizes = [len(v) for v in self._semantic_hash_table.values()]
        median_density = statistics.median(bucket_sizes)
        mean_density = statistics.mean(bucket_sizes)
        min_density = min(bucket_sizes)
        max_density = max(bucket_sizes)
        print('number of buckets:', number_of_buckets)
        print('median density:', median_density)
        print('mean density:', mean_density)
        print('min density:', min_density)
        print('max density:', max_density)

    def search(self, query, threshold=3, top_n=10):
        query_keys = self._encode_items(query)[0]
        item_scores = collections.defaultdict(int)
        for key in query_keys:
            search_scores = self._score_items(key)
            for item_id, score in search_scores.items():
                item_scores[item_id] += score
        return sorted(item_scores, key=lambda x: x[-1], descending=True)[:top_n]

    def _populate_data_structures(self, items):
        codes = self._encode_items(items)
        item_codes = [(item, code) for item, item_codes in zip(items, codes)
                                   for code in item_codes]
        if DEBUG:
            item_codes = tqdm.tqdm(item_codes)
        for item, code in item_codes:
            item_id = id(item)
            self._items[item_id] = item
            self._semantic_hash_table[code].append(item_id)

    def _encode_items(self, items):
        codes = self.encoder.predict(items)
        return codes.reshape(len(codes), -1)

    def _score_items(self, key):
        item_scores = {}
        d = _hamming_distance(key, self._table_keys)
        within_threshold = np.argwhere(d <= self.threshold)
        key_scores = 2**(3-d[within_threshold])
        for score, key in zip(key_scores, self._table_keys[within_threshold]):
            for item in self._semantic_hash_table[key]:
                item_id = id(item)
                item_score = max(score, item_scores.get(item_id, 0))
                item_scores[item_id] = item_score
        return item_scores
