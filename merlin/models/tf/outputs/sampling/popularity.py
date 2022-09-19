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

from merlin.models.tf.outputs.sampling.base import Candidate, CandidateSampler


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PopularityBasedSamplerV2(CandidateSampler):
    """
    Provides a popularity-based negative sampling for the softmax layer
    to ensure training efficiency when the catalog of items is very large.
    The capacity of the queue is fixed and is equal to the catalog size.

    Parameters
    ----------
    max_id: int
        The maximum id value to be sampled. It should be equal to the
        categorical feature cardinality
    min_id: int
        The minimum id value to be sampled. Useful to ignore the first categorical
        encoded ids, which are usually reserved for <nulls>, out-of-vocabulary or padding.
        Defaults to 0.
    max_num_samples: int
        The number of unique negatives to sample at each batch.
    seed: int
        Fix the random values returned by the sampler to ensure reproducibility
        Defaults to None
    """

    def __init__(
        self,
        max_id: int,
        min_id: int = 0,
        max_num_samples: int = 10,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(max_num_samples=max_num_samples, **kwargs)
        self.max_id = max_id
        self.min_id = min_id
        self.seed = seed

        assert (
            self.max_num_samples <= self.max_id
        ), f"Number of items to sample `{self.max_num_samples}`"
        f"should be less than total number of ids `{self.max_id}`"

    def add(self, items: Candidate):
        pass

    def call(
        self,
        positive_items: Candidate = None,
        features=None,
        targets=None,
        training=False,
        testing=False,
    ) -> Candidate:
        return self.sample()

    def sample(self) -> Candidate:
        """
        Method to sample `max_num_samples` unique negatives.
        This implementation does not require the actual item frequencies/probabilities
        distribution, but instead tries to approximate the item
        probabilities using the log_uniform (zipfisan) distribution.
        The only requirement is that the item ids are decreasingly sorted by their count frequency.
        We use the default log-uniform (zipfian) sampler given by Tensorflow:
        [log_uniform_candidate_sampler](https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler)
        We note that the `Categorify` op provided by nvtabular supports
        the frequency-based encoding as default.

        Returns
        -------
        Items
            The negative items ids
        """
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

        return Candidate(sampled_ids, {})

    def get_config(self):
        config = super().get_config()
        config["max_id"] = self.max_id
        config["min_id"] = self.min_id
        config["max_num_samples"] = self.max_num_samples
        config["seed"] = self.seed
        return config
