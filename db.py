import collections
import statistics

class DB:
    def __init__(self, encoder, items=None):
        self.encoder = encoder
        output_dim = self.encoder.output_shape[-1]
        self._init_db(items)

    def search(self, item, threshold=3, top_n=10):
        key = self._make_keys(item)[0]
        hits = self._find_hits(key, threshold)
        items = self._fetch_items(hits, top_n)
        return items

    def _find_hits(self, key, threshold):
        hits = collections.defaultdict(int)
        for other_key in self._db:
            dist = hamming_distance(other_key, key)
            if dist <= threshold:
                hits[other_key] += 2**(threshold-dist)
        return hits

    def _fetch_items(self, hits, top_n):
        items = []
        sorted_hits = sorted(hits.items(), key=lambda x: x[1])
        for key, score in sorted_hits:
            # items from the same bucket are added arbitrarily
            for item in self._db[key]:
                items.append(item)
                if len(items) > top_n:
                    return items
        return items

    def _init_db(self, items):
        self._db = collections.defaultdict(list)
        keys = self._make_keys(items)
        for key, item in zip(keys, items):
            self._db[key].append(item)
        # defaultdict is convenient when initializing the DB instance
        # but dangerous to keep around.
        self._db.default_factory = None

    def _make_keys(self, items):
        codes = self.encoder.predict(items).flatten()
        return codes.astype(np.uint32)
    
    def describe(self):
        bucket_sizes = [len(v) for v in self._db.values()]
        median_density = statistics.median(bucket_sizes)
        mean_density = statistics.mean(bucket_sizes)
        min_density = min(bucket_sizes)
        max_density = max(bucket_sizes)
        print('median density:', median_density)
        print('mean density:', mean_density)
        print('min density:', min_density)
        print('max density:', max_density)
