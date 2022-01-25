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

import abc
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..core import EmbeddingWithMetadata, Tag
from ..layers.queue import FIFOQueue
from ..typing import TabularData


class ItemSampler(abc.ABC, Layer):
    def __init__(
        self,
        max_num_samples: Optional[int] = None,
        **kwargs,
    ):
        super(ItemSampler, self).__init__(**kwargs)
        self.set_max_num_samples(max_num_samples)

    @abc.abstractmethod
    def add(self, embeddings: tf.Tensor, items_metadata: TabularData, training=True):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self) -> EmbeddingWithMetadata:
        raise NotImplementedError()

    def _check_inputs_batch_sizes(self, inputs: TabularData) -> bool:
        embeddings_batch_size = tf.shape(inputs["embeddings"])[0]
        for feat_name in inputs["metadata"]:
            metadata_feat_batch_size = tf.shape(inputs["metadata"][feat_name])[0]

            tf.assert_equal(
                embeddings_batch_size,
                metadata_feat_batch_size,
                "The batch size (first dim) of embeddings "
                f"({int(embeddings_batch_size)}) and metadata "
                f"features ({int(metadata_feat_batch_size)}) must match.",
            )

    @property
    def required_features(self) -> List[str]:
        return []

    @property
    def max_num_samples(self) -> int:
        return self._max_num_samples

    def set_max_num_samples(self, value) -> None:
        self._max_num_samples = value


class InBatchSampler(ItemSampler):
    """Provides in-batch sampling [1] for two-tower item retrieval
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

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def set_batch_size(self, value):
        self._batch_size = value
        if value is not None:
            self.set_max_num_samples(value)

    def build(self, input_shapes: TabularData) -> None:
        if self._batch_size is None:
            self.set_batch_size(input_shapes["embeddings"][0])

    def call(self, inputs: TabularData, training=True) -> EmbeddingWithMetadata:
        """Returns the item embeddings and item metadata from
        the current batch.
        The implementation is very simple, as it just returns the current
        item embeddings and metadata, but it is necessary to have
        `InBatchSampler` under the same interface of other more advanced samplers
        (e.g. `CachedCrossBatchSampler`).

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
        EmbeddingWithMetadata
            Value object with the sampled item embeddings and item metadata
        """
        self.add(inputs, training)
        items_embeddings = self.sample()
        return items_embeddings

    def add(self, inputs: TabularData, training=True) -> None:
        self._check_inputs_batch_sizes(inputs)
        self._last_batch_items_embeddings = inputs["embeddings"]
        self._last_batch_items_metadata = inputs["metadata"]

    def sample(self) -> EmbeddingWithMetadata:
        return EmbeddingWithMetadata(
            self._last_batch_items_embeddings, self._last_batch_items_metadata
        )


class CachedCrossBatchSampler(ItemSampler):
    """Provides efficient cached cross-batch [1] / inter-batch [2] negative sampling
    for two-tower item retrieval model. The caches consists of a fixed capacity FIFO queue
    which keeps the item embeddings from the last N batches. All items in the queue are
    sampled as negatives for upcoming batches.
    It is more efficent than computing embeddings exclusively for negative items.
    This is a popularity-biased sampling as popular items are observed more often
    in training batches.
    Compared to `InBatchSampler`, the `CachedCrossBatchSampler` allows for larger number
    of negative items, not limited to the batch size. The gradients are not computed
    for the cached negative embeddings which is a scalable approach. A common combination
    of samplers for the `ItemRetrievalScorer` is `[InBatchSampler(),
    CachedCrossBatchSampler(ignore_last_batch_on_sample=True)]`, which computes gradients for
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
    capacity: int
        The queue capacity to store samples
    ignore_last_batch_on_sample: bool
        Whether should include the last batch in the sampling. By default `False`,
        as for sampling from the current batch we recommend `InBatchSampler()`, which
        allows computing gradients for in-batch negative items
    """

    def __init__(
        self,
        capacity: int,
        ignore_last_batch_on_sample: bool = True,
        **kwargs,
    ):
        assert capacity > 0
        super().__init__(max_num_samples=capacity, **kwargs)
        self.ignore_last_batch_on_sample = ignore_last_batch_on_sample
        self.item_metadata_dtypes = None

        self._last_batch_size = 0
        self._item_embeddings_queue = None

    def _maybe_build(self, inputs: TabularData) -> None:
        items_metadata = inputs["metadata"]
        if self.item_metadata_dtypes is None:
            self.item_metadata_dtypes = {
                feat_name: items_metadata[feat_name].dtype for feat_name in items_metadata
            }
        super()._maybe_build(inputs)

    def build(self, input_shapes: TabularData) -> None:
        queue_size = self.max_num_samples

        item_embeddings_dims = list(input_shapes["embeddings"][1:])
        self._item_embeddings_queue = FIFOQueue(
            capacity=queue_size,
            dims=item_embeddings_dims,
            dtype=tf.float32,
            name="item_emb",
        )

        self.items_metadata_queue = dict()
        items_metadata = input_shapes["metadata"]
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
                "The CachedCrossBatchSampler layer was not built yet. "
                "You need to call() that layer at least once before "
                "so that it is built before calling add() or sample() directly"
            )

    def call(self, inputs: TabularData, training=True) -> EmbeddingWithMetadata:
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
              The `CachedCrossBatchSampler` does not use metadata features
              specifically, but "item_id" is required when using in combination with
              `ItemRetrievalScorer(..., sampling_downscore_false_negatives=True)`, so that
              false negatives are identified and downscored.
        training : bool, optional
            Flag indicating if on training mode, by default True

        Returns
        -------
        EmbeddingWithMetadata
            Value object with the sampled item embeddings and item metadata
        """
        if self.ignore_last_batch_on_sample:
            items_embeddings = self.sample()
            self.add(inputs, training)
        else:
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

        if training:
            items_embeddings = inputs["embeddings"]
            items_metadata = inputs["metadata"]

            self._item_embeddings_queue.enqueue_many(items_embeddings)
            for feat_name in items_metadata:
                self.items_metadata_queue[feat_name].enqueue_many(items_metadata[feat_name])

            self._last_batch_size = tf.shape(items_embeddings)[0]

    def sample(self) -> EmbeddingWithMetadata:
        self._check_built()
        items_embeddings = self._item_embeddings_queue.list_all()
        items_metadata = {
            feat_name: self.items_metadata_queue[feat_name].list_all()
            for feat_name in self.items_metadata_queue
        }

        return EmbeddingWithMetadata(
            items_embeddings,
            items_metadata,
        )


class CachedUniformSampler(ItemSampler):
    """Provides a cached uniform negative sampling for two-tower item retrieval model.
    It is similar to the `CachedCrossBatchSampler`, the main difference
    is that the `CachedUniformSampler` is a popularity-based sampler and `CachedUniformSampler`
    only keeps unique item embeddings in the queue for uniform sampling.
    The caches consists of two fixed capacity internal FIFO queues that hold both
    item ids and item embeddings.
    It ensures that each item id (and corresponding embedding) is added only once into
    the queue. If the item id was already included in the queue by a previous batch,
    the embedding is updated.
    As the queues reach their capacity of unique items, new items will replace the
    first items added to the queue.

    This is a cached implementation of [1], where those authors proposed combining
    in-batch sampling (our `InBatchSampler()`) with uniform sampling. Differently from
    [1] which requires a separate dataset with the all unique items (and corresponding features)
    to generate the item embeddings, our streaming approach in `CachedUniformSampler` keeps
    caching new items as they appear in the batches. That means that the very first
    processsed batches will have less negative samples.

    P.s. Ignoring the false negatives (negative items equal to the positive ones) is
    managed by `ItemRetrievalScorer(..., sampling_downscore_false_negatives=True)`

    References
    ----------
    [1] Yang, Ji, et al. "Mixed negative sampling for learning two-tower neural networks in
    recommendations." Companion Proceedings of the Web Conference 2020. 2020.

    Parameters
    ----------
    capacity : int
        The maximum number of unique items that can be stored
    """

    def __init__(
        self,
        capacity: int,
        **kwargs,
    ):
        assert capacity > 0
        super().__init__(capacity, **kwargs)

        self._item_embeddings_queue = None
        self._item_ids_queue = None

    def build(self, input_shapes: TabularData) -> None:
        # Reserving additional space for the current batch when ignore_last_batch_on_sample=True
        queue_size = self.max_num_samples

        item_embeddings_dims = list(input_shapes["embeddings"][1:])
        self._item_embeddings_queue = FIFOQueue(
            capacity=queue_size,
            dims=item_embeddings_dims,
            dtype=tf.float32,
            name="item_emb",
        )

        self._item_ids_queue = FIFOQueue(
            capacity=queue_size,
            dims=[],
            dtype=tf.int32,
            name="item_id",
        )

    def _check_built(self) -> None:
        if self._item_embeddings_queue is None:
            raise Exception(
                "The CachedUniformSampler layer was not built yet. "
                "You need to call() that layer at least once before "
                "so that it is built before calling add() or sample() directly"
            )

    def call(self, inputs: TabularData, training=True) -> EmbeddingWithMetadata:
        """Adds the unique items of the current batch into a FIFO queue cache
        and returns all cached unique items, which is equivalent to uniform sampling

        Parameters
        ----------
        inputs : TabularData
            Dict with two keys:
              "items_embeddings": Items embeddings tensor
              "items_metadata": Dict like `{"<feature name>": "<feature tensor>"}` which contains
              features that might be relevant for the sampler (e.g. item id, item popularity, item
              recency).
              The `CachedUniformSampler` required "item_id" to avoid storing repeated items.
        training : bool, optional
            Flag indicating if on training mode, by default True

        Returns
        -------
        EmbeddingWithMetadata
            Value object with the sampled item embeddings and item metadata
        """
        self.add(inputs, training)
        items_embeddings = self.sample()
        return items_embeddings

    def _check_inputs(self, inputs):
        assert (
            str(Tag.ITEM_ID) in inputs["metadata"]
        ), "The 'item_id' metadata feature is required by UniformSampler."

        tf.assert_equal(
            tf.shape(inputs["embeddings"])[0],
            tf.shape(inputs["metadata"][str(Tag.ITEM_ID)])[0],
        )

    def add(
        self,
        inputs: TabularData,
        training: bool = True,
    ) -> None:
        self._check_built()
        self._check_inputs(inputs)

        if training:
            items_embeddings = inputs["embeddings"]
            item_ids = inputs["metadata"][str(Tag.ITEM_ID)]

            # Removing from inputs repeated item ids and corresponding embeddings
            item_ids, items_embeddings = self._get_unique_item_ids_embeddings(
                item_ids, items_embeddings
            )

            # Identifying what are the internal indices of the item ids in its queue
            item_ids_idxs = self._item_ids_queue.index_of(item_ids)
            # Creating masks for new and existing items
            new_items_mask = tf.equal(item_ids_idxs, -1)
            existing_items_mask = tf.logical_not(new_items_mask)

            # Updating embeddings of existing items
            self._item_embeddings_queue.update_by_indices(
                indices=tf.expand_dims(item_ids_idxs[existing_items_mask], -1),
                values=items_embeddings[existing_items_mask],
            )

            # Adding to the queue items not found
            self._item_embeddings_queue.enqueue_many(items_embeddings[new_items_mask])
            self._item_ids_queue.enqueue_many(item_ids[new_items_mask])

    def _get_unique_item_ids_embeddings(self, item_ids, items_embeddings):
        sorting_item_ids_idx = tf.argsort(item_ids)
        sorted_item_ids = tf.gather(item_ids, sorting_item_ids_idx)
        sorted_items_embeddings = tf.gather(items_embeddings, sorting_item_ids_idx)
        non_repeated_item_ids_mask = tf.concat(
            [
                # First element is never repeated
                tf.constant(
                    True,
                    shape=(1,),
                ),
                tf.not_equal(sorted_item_ids[:-1], sorted_item_ids[1:]),
            ],
            axis=0,
        )

        item_ids = sorted_item_ids[non_repeated_item_ids_mask]
        items_embeddings = sorted_items_embeddings[non_repeated_item_ids_mask]

        return item_ids, items_embeddings

    def sample(self) -> EmbeddingWithMetadata:
        self._check_built()
        items_embeddings = self._item_embeddings_queue.list_all()
        items_ids = self._item_ids_queue.list_all()

        return EmbeddingWithMetadata(items_embeddings, metadata={str(Tag.ITEM_ID): items_ids})
