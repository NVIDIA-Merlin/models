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
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.python.ops import embedding_ops

from merlin.models.tf.blocks.sampling.base import EmbeddingWithMetadata, ItemSampler
from merlin.models.tf.blocks.sampling.queue import FIFOQueue
from merlin.models.tf.typing import TabularData


class CachedCrossBatchSampler(ItemSampler):
    """Provides efficient cached cross-batch [1]_ / inter-batch [2]_ negative sampling
    for two-tower item retrieval model. The caches consists of a fixed capacity FIFO queue
    which keeps the item embeddings from the last N batches. All items in the queue are
    sampled as negatives for upcoming batches.
    It is more efficient than computing embeddings exclusively for negative items.
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
        self.item_metadata_dtypes: Dict[str, tf.dtypes.DType] = {}

        self._last_batch_size = 0
        self._item_embeddings_queue: Optional[FIFOQueue] = None

    @property
    def item_embeddings_queue(self) -> FIFOQueue:
        if not self._item_embeddings_queue:
            raise ValueError("Item embeddings queue is not initialized")

        return self._item_embeddings_queue

    def _maybe_build(self, inputs: TabularData) -> None:
        items_metadata = inputs["metadata"]
        if not self.item_metadata_dtypes:
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
            dims = []
            if items_metadata[feat_name].dims is not None:
                dims = list(items_metadata[feat_name][1:])
            self.items_metadata_queue[feat_name] = FIFOQueue(
                capacity=queue_size,
                dims=dims,
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

    def add(  # type: ignore
        self,
        inputs: TabularData,
        training: bool = True,
    ) -> None:
        self._check_built()

        if training:
            self._check_inputs_batch_sizes(inputs)
            items_embeddings = inputs["embeddings"]
            items_metadata = inputs["metadata"]

            self.item_embeddings_queue.enqueue_many(items_embeddings)
            for feat_name in items_metadata:
                self.items_metadata_queue[feat_name].enqueue_many(items_metadata[feat_name])

            self._last_batch_size = tf.shape(items_embeddings)[0]

    def sample(self) -> EmbeddingWithMetadata:
        self._check_built()
        items_embeddings = self.item_embeddings_queue.list_all()
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
    processed batches will have less negative samples.

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

    def add(  # type: ignore
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
            self.item_embeddings_queue.update_by_indices(
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

    This implementation does not require the actual item frequencies/probabilities
    distribution, but instead tries to approximate the item
    probabilities using the log_uniform (zipfian) distribution.
    The only requirement is that the item ids are decreasingly sorted by their count frequency.
    We use the default log-uniform (zipfian) sampler given by Tensorflow:
        [log_uniform_candidate_sampler](https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler)
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
        seed: Optional[int] = None,
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
            self.max_id + 1,
            "The first dimension of the items embeddings "
            f"({int(tf.shape(item_weights)[0])}) and "
            f"the the number of possible classes ({self.max_id+1}) should match.",
        )

        items_embeddings = self.sample(item_weights)
        return items_embeddings

    def _required_features(self):
        return [self.item_id_feature_name]

    def sample(self, item_weights) -> EmbeddingWithMetadata:  # type: ignore
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

    def get_distribution_probs(self):
        """Tries to approximate the log uniform (zipfian) distribution
        used by tf.random.log_uniform_candidate_sampler
        (https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler)

        Returns
        -------
        tf.Tensor
            A tensor with the expected probability distribution of item ids
            assuming log-uniform (zipfian) distribution
        """
        range_max = self.max_id - self.min_id
        ids = tf.range(0, range_max, dtype=tf.float32)
        estimated_probs = (tf.math.log(ids + 2.0) - tf.math.log(ids + 1.0)) / tf.math.log(
            range_max + 1.0
        )
        # Appending zero(s) in the beginning as padding items should never be samples
        # (thus prob must be zero)
        estimated_probs = tf.concat(
            [tf.zeros(self.min_id + 1, dtype=tf.float32), estimated_probs], axis=0
        )
        return estimated_probs
