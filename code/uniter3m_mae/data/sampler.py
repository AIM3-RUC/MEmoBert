"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

sampler for length bucketing (batch by tokens)
Jinming Update: the batch-size is the real batch-szie, not the max-token
"""
import random

from torch.utils.data import Sampler
from cytoolz import partition_all

class TokenBucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, size_multiple=4):
        self._lens = lens
        self.batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._size_mul = size_multiple

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        ids = self._create_ids()
        random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = []
        for bucket in buckets:
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                if len(batch_indices) >= self.batch_size:
                    assert len(batch_indices) % self._size_mul == 0
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        # print('[Debug] {} batches'.format(len(batches)))
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")
