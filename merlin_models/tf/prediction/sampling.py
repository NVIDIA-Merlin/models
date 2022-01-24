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

import tensorflow as tf

from ..core import Block, ItemSampler, ItemSamplerData
from ..typing import TabularData
from ..utils.tf_utils import FIFOQueue


class InBatchSampler(ItemSampler):
    """Provides in-batch sampling [1] for two-tower item retrieval
    models. The implementation is very simple, as it
    just returns the current item embeddings and metadata, but it is necessary to have
    `InBatchSampler` under the same interface of other more advanced samplers
    (e.g. `CachedBatchesSampler`).
    In a nutshell, for a given (user,item) embeddings pair, the other in-batch item
    embeddings are used as negative items, rather than computing different embeddings
    exclusively for negative items.
    This is a popularity-biased sampling     as popular items are observed more often
    in training batches.
    P.s. Ignoring the false negatives (negative items equal to the positive ones) is
    managed by `ItemRetrievalScorer(..., sampling_downscore_false_negatives=True)`

    References
    ----------
    [1] Yi, Xinyang, et al. "Sampling-bias-corrected neural modeling for large corpus item
    recommendations." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.

    Parameters
    ----------
    batch_size : int, optional
        The batch size. If not set it is inferred when the layer is built (first call())
    """

    def __init__(self, batch_size: int = None, **kwargs):
        super().__init__(max_num_samples=batch_size, **kwargs)
        self._last_batch_items_embeddings: tf.Tensor = None
        self._last_batch_items_metadata: TabularData = None
        self.set_batch_size(batch_size)

    def build(self, input_shapes: TabularData) -> None:
        if self._batch_size is None:
            self.set_batch_size(input_shapes["items_embeddings"][0])

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
        self._max_num_samples = batch_size

    def call(self, inputs: TabularData, training=True) -> ItemSamplerData:
        """Returns the item embeddings and item metadata from
        the current batch.
        The implementation is very simple, as it just returns the current
        item embeddings and metadata, but it is necessary to have
        `InBatchSampler` under the same interface of other more advanced samplers
        (e.g. `CachedBatchesSampler`).

        Parameters
        ----------
        inputs : TabularData
            Dict with two keys:
              "items_embeddings": Items embeddings tensor
              "items_metadata": Dict like `{"<feature name>": "<feature tensor>"}` which contains
              features that might be relevant for the sampler.
              The `InBatchSampler` does not use metadata features
              specifically, but "item_id" is required when using in combination with
              `ItemRetrievalScorer(..., sampling_downscore_false_negatives=True)`, so that
              false negatives are identified and downscored.
        training : bool, optional
            Flag indicating if on training mode, by default True

        Returns
        -------
        ItemSamplerData
            Value object with the sampled item embeddings and item metadata
        """
        self.add(inputs, training)
        items_embeddings = self.sample()
        return items_embeddings

    def add(self, inputs: TabularData, training=True) -> None:
        self._check_inputs_batch_sizes(inputs)
        self._last_batch_items_embeddings = inputs["items_embeddings"]
        self._last_batch_items_metadata = inputs["items_metadata"]

    def sample(self) -> ItemSamplerData:
        return ItemSamplerData(self._last_batch_items_embeddings, self._last_batch_items_metadata)


class CachedBatchesSampler(ItemSampler):
    """Provides efficient cached cross-batch [1] / inter-batch [2] negative sampling
    for two-tower item retrieval model. The caches consists of a fixed capacity FIFO queue
    which keeps the item embeddings from the last N batches. All items in the queue are
    sampled as negatives for upcoming batches.
    It is more efficent than computing embeddings exclusively for negative items.
    This is a popularity-biased sampling as popular items are observed more often
    in training batches.
    Compared to `InBatchSampler`, the `CachedBatchesSampler` allows for larger number
    of negative items, not limited to the batch size. The gradients are not computed
    for the cached negative embeddings which is a scalable approach. A common combination
    of samplers for the `ItemRetrievalScorer` is `[InBatchSampler(),
    CachedBatchesSampler(ignore_last_batch_on_sample=True)]`, which computes gradients for
    the in-batch negatives and not for the cached item embeddings.
    P.s. Ignoring the false negatives (negative items equal to the positive ones) is
    managed by `ItemRetrievalScorer(..., sampling_downscore_false_negatives=True)`

    References
    ----------
    [1] Wang, Jinpeng, Jieming Zhu, and Xiuqiang He. "Cross-Batch Negative Sampling
    for Training Two-Tower Recommenders." Proceedings of the 44th International ACM
    SIGIR Conference on Research and Development in Information Retrieval. 2021.

    [2] Zhou, Chang, et al. "Contrastive learning for debiased candidate generation
    in large-scale recommender systems." Proceedings of the 27th ACM SIGKDD Conference
    on Knowledge Discovery & Data Mining. 2021.

    Parameters
    ----------
    batch_size : int
        The batch size, which is required to define the sampler capacity
        (batch_size * num_batches_to_cache)
    num_batches_to_cache: int
        The number of batches to cache, which is required to define the sampler capacity
        (batch_size * num_batches_to_cache), defaults to 1.
    ignore_last_batch_on_sample: bool
        Whether should include the last batch in the sampling. By default `False`,
        as for sampling from the current batch we recommend `InBatchSampler()`, which
        allows computing gradients for in-batch negative items
    """

    def __init__(
        self,
        batch_size: int,
        num_batches_to_cache: int = 1,
        post: Optional[Block] = None,
        ignore_last_batch_on_sample: bool = True,
        **kwargs,
    ):
        assert batch_size > 0
        assert num_batches_to_cache > 0

        max_num_samples = batch_size * num_batches_to_cache
        super().__init__(max_num_samples, **kwargs)

        self.batch_size = batch_size
        self.post = post
        self.ignore_last_batch_on_sample = ignore_last_batch_on_sample
        self.item_metadata_dtypes = None

        self._last_batch_size = 0
        self._item_embeddings_queue = None

    def _maybe_build(self, inputs: TabularData) -> None:
        items_metadata = inputs["items_metadata"]
        if self.item_metadata_dtypes is None:
            self.item_metadata_dtypes = {
                feat_name: items_metadata[feat_name].dtype for feat_name in items_metadata
            }
        super()._maybe_build(inputs)

    def build(self, input_shapes: TabularData) -> None:
        # Reserving additional space for the current batch when ignore_last_batch_on_sample=True
        queue_size = self.max_num_samples
        if self.ignore_last_batch_on_sample:
            queue_size += self.batch_size

        item_embeddings_dims = list(input_shapes["items_embeddings"][1:])
        self._item_embeddings_queue = FIFOQueue(
            capacity=queue_size,
            dims=item_embeddings_dims,
            dtype=tf.float32,
            name="item_emb",
        )

        self.items_metadata_queue = dict()
        items_metadata = input_shapes["items_metadata"]
        for feat_name in items_metadata:
            self.items_metadata_queue[feat_name] = FIFOQueue(
                capacity=queue_size,
                dims=list(items_metadata[feat_name][1:]),
                dtype=self.item_metadata_dtypes[feat_name],
                name=f"item_metadata_{feat_name}",
            )

    def _check_built(self) -> None:
        if self._item_embeddings_queue is None:
            raise Exception(
                "The CachedBatchesSampler layer was not built yet. "
                "You need to call() that layer at least once before "
                "so that it is built before calling add() or sample() directly"
            )

    def call(self, inputs: TabularData, training=True) -> ItemSamplerData:
        """Adds the current batch to the FIFO queue cache and samples all items
        embeddings from the last N cached batches.

        Parameters
        ----------
        inputs : TabularData
            Dict with two keys:
              "items_embeddings": Items embeddings tensor
              "items_metadata": Dict like `{"<feature name>": "<feature tensor>"}` which contains
              features that might be relevant for the sampler (e.g. item id, item popularity, item
              recency).
              The `CachedBatchesSampler` does not use metadata features
              specifically, but "item_id" is required when using in combination with
              `ItemRetrievalScorer(..., sampling_downscore_false_negatives=True)`, so that
              false negatives are identified and downscored.
        training : bool, optional
            Flag indicating if on training mode, by default True

        Returns
        -------
        ItemSamplerData
            Value object with the sampled item embeddings and item metadata
        """

        self.add(inputs, training)
        items_embeddings = self.sample()
        return items_embeddings

    def add(
        self,
        inputs: TabularData,
        training: bool = True,
    ) -> None:
        self._check_built()
        self._check_inputs_batch_sizes(inputs)

        items_embeddings = inputs["items_embeddings"]
        items_metadata = inputs["items_metadata"]
        if training:
            self._item_embeddings_queue.enqueue_many(items_embeddings)
            for feat_name in items_metadata:
                self.items_metadata_queue[feat_name].enqueue_many(items_metadata[feat_name])

            self._last_batch_size = tf.shape(items_embeddings)[0]

    def sample(self) -> ItemSamplerData:
        self._check_built()
        items_embeddings = self._item_embeddings_queue.list_all()
        items_metadata = {
            feat_name: self.items_metadata_queue[feat_name].list_all()
            for feat_name in self.items_metadata_queue
        }

        # Checks if should ignore the last batch, assuming the current batch was already added
        # to the sampler queue and that InBatchSampler(stop_gradients=False) will
        # take care of the current batch negatives
        if self.ignore_last_batch_on_sample:
            items_embeddings = items_embeddings[: -self._last_batch_size]
            items_metadata = {
                feat_name: items_metadata[feat_name][: -self._last_batch_size]
                for feat_name in items_metadata
            }

        if self.post is not None:
            items_embeddings = self.post(items_embeddings)

        return ItemSamplerData(
            items_embeddings,
            items_metadata,
        )


class UniformSampler(ItemSampler):
    def __init__(self, stop_gradient: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.stop_gradient = stop_gradient

    def add(self, items_embeddings: tf.Tensor, items_metadata: TabularData, training=True):
        raise NotImplementedError()

    def sample(self) -> TabularData:
        raise NotImplementedError()
