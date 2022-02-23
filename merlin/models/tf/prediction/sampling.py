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
from tensorflow.python.ops import embedding_ops

from merlin.models.tf.blocks.queue import FIFOQueue

from ..core import EmbeddingWithMetadata
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
    """Provides efficient cached cross-batch [1]_ / inter-batch [2]_ negative sampling
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
    .. [1] Wang, Jinpeng, Jieming Zhu, and Xiuqiang He. "Cross-Batch Negative Sampling
       for Training Two-Tower Recommenders." Proceedings of the 44th International ACM
       SIGIR Conference on Research and Development in Information Retrieval. 2021.

    .. [2] Zhou, Chang, et al. "Contrastive learning for debiased candidate generation
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
                f"The {self.__class__.__name__} layer was not built yet. "
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

        if training:
            self._check_inputs_batch_sizes(inputs)
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


class CachedUniformSampler(CachedCrossBatchSampler):
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

    This is a cached implementation of [1]_, where those authors proposed combining
    in-batch sampling (our `InBatchSampler()`) with uniform sampling. Differently from
    [1] which requires a separate dataset with the all unique items (and corresponding features)
    to generate the item embeddings, our streaming approach in `CachedUniformSampler` keeps
    caching new items as they appear in the batches. That means that the very first
    processsed batches will have less negative samples.

    P.s. Ignoring the false negatives (negative items equal to the positive ones) is
    managed by `ItemRetrievalScorer(..., sampling_downscore_false_negatives=True)`

    References
    ----------
    .. [1] Yang, Ji, et al. "Mixed negative sampling for learning two-tower neural networks in
       recommendations." Companion Proceedings of the Web Conference 2020. 2020.

    Parameters
    ----------
    capacity: int
        The queue capacity to store samples
    ignore_last_batch_on_sample: bool
        Whether should include the last batch in the sampling. By default `False`,
        as for sampling from the current batch we recommend `InBatchSampler()`, which
        allows computing gradients for in-batch negative items
    item_id_feature_name: str
        Name of the column containing the item ids
        Defaults to `item_id`
    """

    def __init__(
        self,
        capacity: int,
        ignore_last_batch_on_sample: bool = True,
        item_id_feature_name: str = "item_id",
        **kwargs,
    ):
        super().__init__(
            capacity=capacity, ignore_last_batch_on_sample=ignore_last_batch_on_sample, **kwargs
        )
        self.item_id_feature_name = item_id_feature_name

    def _check_inputs(self, inputs):
        assert (
            str(self.item_id_feature_name) in inputs["metadata"]
        ), "The 'item_id' metadata feature is required by UniformSampler."

    def add(
        self,
        inputs: TabularData,
        training: bool = True,
    ) -> None:
        """Updates the FIFO queue with batch item embeddings (for items whose ids were
        already added to the queue) and adds to the queue the items seen for the first time

        Parameters
        ----------
        inputs : TabularData
            Dict with two keys:
              "items_embeddings": Items embeddings tensor
              "items_metadata": Dict like `{"<feature name>": "<feature tensor>"}` which contains
              features that might be relevant for the sampler (e.g. item id, item popularity, item
              recency).
              The `CachedUniformSampler` requires the "item_id" feature to identify items already
              added to the queue.
        training : bool, optional
            Flag indicating if on training mode, by default True
        """

        self._check_built()

        if training:
            self._check_inputs(inputs)

            # Removing from inputs repeated item ids and corresponding embeddings
            unique_items = self._get_unique_items(inputs["embeddings"], inputs["metadata"])
            unique_items_ids = unique_items.metadata[self.item_id_feature_name]

            # Identifying what are the internal indices of the item ids in its queue
            item_ids_idxs = self.items_metadata_queue[self.item_id_feature_name].index_of(
                unique_items_ids
            )
            # Creating masks for new and existing items
            new_items_mask = tf.equal(item_ids_idxs, -1)
            existing_items_mask = tf.logical_not(new_items_mask)

            update_indices = tf.expand_dims(item_ids_idxs[existing_items_mask], -1)
            # Updating embeddings of existing items
            self._item_embeddings_queue.update_by_indices(
                indices=update_indices,
                values=unique_items.embeddings[existing_items_mask],
            )

            for feat_name in self.items_metadata_queue:
                self.items_metadata_queue[feat_name].update_by_indices(
                    indices=update_indices,
                    values=unique_items.metadata[feat_name][existing_items_mask],
                )

            # Adding to the queue items not found
            new_items = EmbeddingWithMetadata(
                embeddings=unique_items.embeddings[new_items_mask],
                metadata={
                    feat_name: unique_items.metadata[feat_name][new_items_mask]
                    for feat_name in unique_items.metadata
                },
            )
            super().add(new_items.__dict__, training)

    def _get_unique_items(
        self, items_embeddings: tf.Tensor, items_metadata: TabularData
    ) -> EmbeddingWithMetadata:
        """Extracts the embeddings and corresponding metadata features for
        the unique items found in the batch, based on the item ids

        Parameters
        ----------
        items_embeddings : items_metadata
            Batch item embeddings
        items_metadata : TabularData
            Batch item metadata features

        Returns
        -------
        EmbeddingWithMetadata
            An EmbeddingWithMetadata object with the embeddings and
            metadata features of the unique items
        """
        item_ids = items_metadata[self.item_id_feature_name]
        sorting_item_ids_idx = tf.argsort(item_ids)

        sorted_embeddings = tf.gather(items_embeddings, sorting_item_ids_idx)
        sorted_metadata = {
            feat_name: tf.gather(items_metadata[feat_name], sorting_item_ids_idx)
            for feat_name in items_metadata
        }

        sorted_item_ids = tf.gather(item_ids, sorting_item_ids_idx)

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

        output_embeddings = sorted_embeddings[non_repeated_item_ids_mask]
        output_metadata = {
            feat_name: items_metadata[feat_name][non_repeated_item_ids_mask]
            for feat_name in sorted_metadata
        }

        return EmbeddingWithMetadata(
            embeddings=output_embeddings,
            metadata=output_metadata,
        )


class PopularityBasedSampler(ItemSampler):
    """
    Provides a popularity-based negative sampling for the softmax layer
    to ensure training efficiency when the catalog of items is very large.
    The capacity of the queue is fixed and is equal to the catalog size.
    For each batch, we sample `max_num_samples` unique negatives.

    We use the default log-uniform sampler given by tensorflow:
        [log_uniform_candidate_sampler](https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler)

    We note that this default sampler requires that
    item-ids are encoded based on a decreasing order of their count frequency
    and that the classes' expected counts are approximated based on their index order.
    The `Categorify` op provided by nvtabular supports the frequency-based encoding as default.

    P.s. Ignoring the false negatives (negative items equal to the positive ones) is
    managed by `ItemRetrievalScorer(..., sampling_downscore_false_negatives=True)`

    Parameters
    ----------
    max_num_samples: int
        The number of unique negatives to sample at each batch.
    max_id: int
        The maximum id value to be sampled. It should be equal to the
        categorical feature cardinality
    min_id: int
        The minimum id value to be sampled. Useful to ignore the first categorical
        encoded ids, which are usually reserved for <nulls>, out-of-vocabulary or padding.
        Defaults to 0.
    seed: int
        Fix the random values returned by the sampler to ensure reproducibility
        Defaults to None
    item_id_feature_name: str
        Name of the column containing the item ids
        Defaults to `item_id`
    """

    def __init__(
        self,
        max_id: int,
        min_id: int = 0,
        max_num_samples: int = 100,
        seed: int = None,
        item_id_feature_name: str = "item_id",
        **kwargs,
    ):
        super().__init__(max_num_samples=max_num_samples, **kwargs)
        self.max_id = max_id
        self.min_id = min_id
        self.seed = seed
        self.item_id_feature_name = item_id_feature_name

        assert (
            self.max_num_samples <= self.max_id
        ), f"Number of items to sample `{self.max_num_samples}`"
        f" should be less than total number of ids `{self.max_id}`"

    def _check_inputs(self, inputs):
        assert (
            self.item_id_feature_name in inputs["metadata"]
        ), "The 'item_id' metadata feature is required by PopularityBasedSampler."

    def add(self, embeddings: tf.Tensor, items_metadata: TabularData, training=True):
        pass

    def call(
        self, inputs: TabularData, item_weights: tf.Tensor, training=True
    ) -> EmbeddingWithMetadata:
        if training:
            self._check_inputs(inputs)

        tf.assert_equal(
            int(tf.shape(item_weights)[0]),
            self.max_id,
            "The first dimension of the items embeddings "
            f"({int(tf.shape(item_weights)[0])}) and "
            f"the the number of possible classes ({self.max_id}) should match.",
        )

        items_embeddings = self.sample(item_weights)
        return items_embeddings

    def _required_features(self):
        return [self.item_id_feature_name]

    def sample(self, item_weights) -> EmbeddingWithMetadata:
        sampled_ids, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=tf.ones((1, 1), dtype=tf.int64),
            num_true=1,
            num_sampled=self.max_num_samples,
            unique=True,
            range_max=self.max_id - self.min_id,
            seed=self.seed,
        )

        # Shifting the sampled ids to ignore the first ids (usually reserved for nulls, OOV)
        sampled_ids += self.min_id

        items_embeddings = embedding_ops.embedding_lookup(item_weights, sampled_ids)

        return EmbeddingWithMetadata(
            items_embeddings,
            metadata={self.item_id_feature_name: tf.cast(sampled_ids, tf.int32)},
        )
