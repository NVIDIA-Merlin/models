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
from tensorflow.python.ops import embedding_ops

from merlin.models.tf.blocks.sampling.base import EmbeddingWithMetadata, ItemSampler
from merlin.models.tf.typing import TabularData


@tf.keras.utils.register_keras_serializable(package="merlin.models")
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

    def get_config(self):
        config = super().get_config()
        config["max_id"] = self.max_id
        config["min_id"] = self.min_id
        config["max_num_samples"] = self.max_num_samples
        config["seed"] = self.seed
        config["item_id_feature_name"] = self.item_id_feature_name

        return config
