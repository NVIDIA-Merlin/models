#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Optional

from merlin.models.tf.blocks.sampling.base import ItemSampler, Items


class InBatchSampler(ItemSampler):
    """Provides in-batch sampling [1]_ for two-tower item retrieval
    models. The implementation is very simple, as it
    just returns the current item embeddings and metadata, but it is necessary to have
    `InBatchSampler` under the same interface of other more advanced samplers
    (e.g. `CachedCrossBatchSampler`).
    In a nutshell, for a given (user,item) embeddings pair, the other in-batch item
    embeddings are used as negative items, rather than computing different embeddings
    exclusively for negative items.
    This is a popularity-biased sampling     as popular items are observed more often
    in training batches.
    P.s. Ignoring the false negatives (negative items equal to the positive ones) is
    managed by `ItemRetrievalScorer(..., sampling_downscore_false_negatives=True)`

    References
    ----------
    .. [1] Yi, Xinyang, et al. "Sampling-bias-corrected neural modeling for large corpus item
       recommendations." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.

    Parameters
    ----------
    batch_size : int, optional
        The batch size. If not set it is inferred when the layer is built (first call())
    """

    def __init__(self, batch_size: Optional[int] = None, **kwargs):
        super().__init__(max_num_samples=batch_size, **kwargs)
        self._last_batch: Optional[Items] = None  # type: ignore
        self.set_batch_size(batch_size)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def set_batch_size(self, value):
        self._batch_size = value
        if value is not None:
            self.set_max_num_samples(value)

    def build(self, items: Items) -> None:
        if self._batch_size is None:
            self.set_batch_size(items.ids[0])

    def add(self, items: Items):
        self._check_inputs_batch_sizes(items)
        self._last_batch = items

    def sample(self) -> Items:
        return self._last_batch
