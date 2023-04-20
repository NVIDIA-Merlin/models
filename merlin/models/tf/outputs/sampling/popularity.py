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
    Provides a popularity-based negative sampling for sampled softmax [1]_ [2]_.
    to ensure training efficiency when the catalog of items is very large.
    Items are sampled from the whole catalog. It also allows saving
    the sampling probabilities for both positive and negative candidates,
    that are required by the logQ sampling correction of sampled softmax.
    This class do not require the actual frequency of items. It assumes that
    item ids are sorted by frequency and follow a long tail distribution and
    uses tf.random.log_uniform_candidate_sampler() for sampling the candidate ids.

    References
    ----------
    .. [1] Yoshua Bengio and Jean-Sébastien Sénécal. 2003. Quick Training of Probabilistic
       Neural Nets by Importance Sampling. In Proceedings of the conference on Artificial
       Intelligence and Statistics (AISTATS).

    .. [2] Y. Bengio and J. S. Senecal. 2008. Adaptive Importance Sampling to Accelerate
       Training of a Neural Probabilistic Language Model. Trans. Neur. Netw. 19, 4 (April
       2008), 713–722. https://doi.org/10.1109/TNN.2007.912312


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
    unique: True
        Whether to return unique candidate ids or allow for repeated ones
    seed: int
        Fix the random values returned by the sampler to ensure reproducibility
        Defaults to None
    """

    def __init__(
        self,
        max_id: int,
        min_id: int = 0,
        max_num_samples: int = 10,
        unique: Optional[bool] = True,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(max_num_samples=max_num_samples, **kwargs)
        self.max_id = max_id
        self.min_id = min_id
        self.seed = seed
        self.unique = unique

        self.sampling_dist = self.get_sampling_distribution()

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
        (
            sampled_ids,
            _,
            _,
        ) = tf.random.log_uniform_candidate_sampler(
            # This is just a placeholder for true_classestrue classes.
            # It should be provided the positive ids here if wanted to
            # get the expected count probs returned.
            # We rather make usage of CandidateSampler.with_sampling_probs()
            # method to get the sampling probs from positives and negatives
            true_classes=tf.ones((1, 1), dtype=tf.int64),
            num_true=1,
            num_sampled=self.max_num_samples,
            unique=self.unique,
            range_max=self.max_id - self.min_id,
            seed=self.seed,
        )
        # Shifting the sampled ids to ignore the first ids (usually reserved for nulls, OOV)
        sampled_ids += self.min_id
        sampled_ids = tf.expand_dims(sampled_ids, -1)

        sampled_ids = tf.stop_gradient(sampled_ids)

        return Candidate(id=sampled_ids, metadata={})

    def get_sampling_distribution(self) -> tf.Tensor:
        """Returns the approximated distribution used to sample items
        by using tf.random.log_uniform_candidate_sampler()

        Returns
        -------
        tf.Tensor
            Probabilities of each item to be sampled
        """
        log_indices = tf.math.log(tf.range(1.0, self.max_id - self.min_id + 2.0, 1.0))
        sampling_probs = (log_indices[1:] - log_indices[:-1]) / log_indices[-1]

        if self.unique:
            # Below is a more numerically stable implementation of the probability of
            # sampling an item at least once (suitable for sampling unique items)
            # P(item is sampled at least once) = 1 - P(item is not sampled)^num_trials
            # where P(item is not sampled) = 1-p and p is the
            # probability to be sampled
            sampling_probs = -tf.math.expm1(self.max_num_samples * tf.math.log1p(-sampling_probs))

        # Shifting probs if first values of item id mapping table are reserved
        if self.min_id > 0:
            sampling_probs = tf.concat(
                [tf.zeros([self.min_id], dtype=sampling_probs.dtype), sampling_probs], axis=0
            )

        sampling_probs = tf.stop_gradient(sampling_probs)

        return sampling_probs

    def with_sampling_probs(self, items: Candidate) -> Candidate:
        """Returns a copy of the Candidate named tuple with
        the sampling_probs set,

        Parameters
        ----------
        items : Candidate
            Positive or negative candidate items

        Returns
        -------
        Candidate
            Candidate items with sampling probability set
        """
        sampling_probs = tf.gather(self.sampling_dist, items.id)
        items_with_sampling_prob = items.with_sampling_prob(sampling_probs)
        return items_with_sampling_prob

    def get_config(self):
        config = super().get_config()
        config["max_id"] = self.max_id
        config["min_id"] = self.min_id
        config["max_num_samples"] = self.max_num_samples
        config["seed"] = self.seed
        return config
