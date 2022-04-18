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

# Adapted from source code: https://github.com/karlhigley/ranking-metrics-torch
from typing import Optional, Sequence, Union

import tensorflow as tf
from keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.keras import backend
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import get as get_metric

from merlin.models.tf.metrics import metrics_registry
from merlin.models.tf.utils.tf_utils import get_candidate_probs

MIN_FLOAT = 1e-16
METRIC_PARAMETERS_DOCSTRING = """
    predicted_candidates_probs: tf.Tensor
        A tensor with shape (batch_size, n_items) corresponding to
        the sorted predicted candidate probabilities. Where the item probability is
        derived from the frequency of the items in the training set as follow:
        `item_prob = item_freq_count / sum(item_freq_count)`
    k : int
        The cut-off for popularity metrics
"""


def novelty_at(
    predicted_candidates_probs: tf.Tensor,
    k: int = 5,
) -> tf.Tensor:
    """
    Computes novely@K metric
    ----------
    {METRIC_PARAMETERS_DOCSTRING}
    """
    return -tf.math.log(predicted_candidates_probs[:, :k] + MIN_FLOAT)


def popularity_bias_at(
    predicted_candidates_probs: tf.Tensor,
    k: int = 5,
) -> tf.Tensor:
    """
    Computes popularity bias@K metric
    ----------
    {METRIC_PARAMETERS_DOCSTRING}
    """
    return tf.reduce_sum(predicted_candidates_probs[:, :k], -1) / k


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class PopularityMetric(Mean):
    """
    Parameters
    ----------
    fn:
        The popularity metric function to wrap,
        with signature `fn(predicted_candidates_probs, k, **kwargs)`.

    item_freq_probs: Union[tf.Tensor, Sequence]
        A Tensor or list with item frequencies (if is_prob_distribution=False)
        or with item probabilities (if is_prob_distribution=True)
    is_prob_distribution: bool, optional
        If True, the item_freq_probs should be a probability distribution of the items.
        If False, the item frequencies is converted to probabilities, by default True
    k: int, optional
        The cut-off for popularity metrics, by default 5
    """

    def __init__(
        self,
        fn,
        item_freq_probs: Union[tf.Tensor, Sequence],
        k=5,
        is_prob_distribution: bool = True,
        name: str = None,
        dtype=None,
        **kwargs,
    ):
        if name is not None:
            name = f"{name}_{k}"
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self.k = k
        self._fn_kwargs = kwargs
        self.is_prob_distribution = is_prob_distribution

        candidate_probs = get_candidate_probs(item_freq_probs, self.is_prob_distribution)
        self.candidate_probs = tf.Variable(
            candidate_probs,
            name="candidate_probs",
            trainable=False,
            dtype=tf.float32,
            validate_shape=False,
            shape=tf.shape(candidate_probs),
        )

    def update_state(
        self,
        y_pred: tf.Tensor,
        y_true: Optional[tf.Tensor] = None,
        sample_weight: Optional[tf.Tensor] = None,
    ):
        """Updates the state of the popularity metric computed by `self._fn`.
        Parameters
        ----------
        y_true : tf.Tensor
            A tensor with shape (batch_size, 1) corresponding to the true labels ids.
        y_pred : tf.Tensor
            A tensor with shape (batch_size, n_items) corresponding to
            sorted predicted item ids.
        sample_weight : Optional[tf.Tensor], optional
            Optional array of the same length as y_pred,
            containing weights to apply to the model's loss for each sample.

        """
        y_true, y_pred = self.check_cast_inputs(y_true, y_pred)
        predicted_probs = tf.gather(self.candidate_probs, y_pred)

        ag_fn = tf.__internal__.autograph.tf_convert(
            self._fn, tf.__internal__.autograph.control_status_ctx()
        )

        matches = ag_fn(
            predicted_candidates_probs=predicted_probs,
            k=self.k,
            **self._fn_kwargs,
        )
        return super().update_state(matches, sample_weight=sample_weight)

    def check_cast_inputs(self, labels, predictions):
        tf.assert_equal(
            tf.rank(predictions), 2, f"predictions must be 2-D tensor (got {predictions.shape})"
        )
        if labels is not None:
            tf.assert_equal(tf.rank(labels), 2, f"labels must be 2-D tensor (got {labels.shape})")
            labels = tf.cast(labels, tf.int32)
        return labels, tf.cast(predictions, tf.int32)

    def get_config(self):
        config = {}
        if type(self) is PopularityMetric:
            # Only include function argument when the object is of a subclass.
            config["fn"] = self._fn
            config["k"] = self.k
            config["is_prob_distribution"] = self._fn_kwargs.get("is_prob_distribution", True)

            for k, v in self._fn_kwargs.items():
                config[k] = backend.eval(v) if is_tensor_or_variable(v) else v
            base_config = super(PopularityMetric, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        fn = config.pop("fn", None)
        k = config.pop("k", None)
        is_prob_distribution = config.pop("is_prob_distribution", None)
        if cls is PopularityMetric:
            return cls(get_metric(fn), k=k, is_prob_distribution=is_prob_distribution, **config)
        return super(PopularityMetric, cls).from_config(config)


@metrics_registry.register_with_multiple_names("novelty_at", "novelty")
class NoveltyAt(PopularityMetric):
    def __init__(self, item_freq_probs, k=10, is_prob_distribution=False, name="novelty"):
        super().__init__(
            novelty_at, item_freq_probs, k=k, is_prob_distribution=is_prob_distribution, name=name
        )


@metrics_registry.register_with_multiple_names("popularity_bias_at", "popularity_bias")
class PopularityBiasAt(PopularityMetric):
    def __init__(self, item_freq_probs, k=10, is_prob_distribution=False, name="novelty"):
        super().__init__(
            popularity_bias_at,
            item_freq_probs,
            k=k,
            is_prob_distribution=is_prob_distribution,
            name=name,
        )


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ItemCoverageAt(tf.keras.metrics.Metric):
    """Computes the prediction coverage for a top-k recommendation model.

    Parameters
    ----------
    num_unique_items : int
        The number of unique items in the train set.
    k : int, optional
        The cut-off for coverage metric, by default 5

    Returns
    ----------
    coverage: float
        The prediction coverage of the recommendations at the k-th position.
    """

    def __init__(self, num_unique_items: int, k=5, name: str = None, dtype=None, **kwargs):
        if name is not None:
            name = f"{name}_{k}"
        super().__init__(name=name, dtype=dtype)

        self.k = k
        self.num_unique_items = num_unique_items

        self.predicted_items_count = tf.Variable(
            tf.zeros((num_unique_items,), dtype=tf.float32),
            name="predicted_items_count",
            trainable=False,
            dtype=tf.float32,
            validate_shape=False,
            shape=(num_unique_items,),
        )

    def update_state(
        self,
        y_pred: tf.Tensor,
        y_true: Optional[tf.Tensor] = None,
        sample_weight: Optional[tf.Tensor] = None,
    ):
        """Updates the state of the item coverage metric by incrementing the count of
        the top-k predicted ids.

        Parameters
        ----------
        y_true : tf.Tensor
            A tensor with shape (batch_size, 1) corresponding to the true labels ids.
        y_pred : tf.Tensor
            A tensor with shape (batch_size, n_items) corresponding to
            sorted predicted item ids. 
        sample_weight : Optional[tf.Tensor], optional
            Optional array of the same length as y_pred,
            containing weights to apply to the model's loss for each sample.

        """
        y_true, y_pred = self.check_cast_inputs(y_true, y_pred)
        unique_predicted_items, _ = tf.unique(tf.reshape(y_pred[:, : self.k], (-1,)))
        self.predicted_items_count = tf.tensor_scatter_nd_add(
            self.predicted_items_count,
            indices=tf.expand_dims(unique_predicted_items, -1),
            updates=tf.ones_like(unique_predicted_items, dtype=tf.float32),
        )

    def result(self):
        coverage = tf.cast(self.predicted_items_count > 0, tf.float32) / self.num_unique_items
        return tf.reduce_sum(coverage)

    def reset_states(self):
        self.predicted_items_count.assign(tf.zeros((self.num_unique_items,), dtype=tf.float32))

    def check_cast_inputs(self, labels, predictions):
        tf.assert_equal(
            tf.rank(predictions), 2, f"predictions must be 2-D tensor (got {predictions.shape})"
        )
        if labels is not None:
            tf.assert_equal(tf.rank(labels), 2, f"labels must be 2-D tensor (got {labels.shape})")
            labels = tf.cast(labels, tf.int32)
        return labels, tf.cast(predictions, tf.int32)

    def get_config(self):
        config = {}
        config["num_unique_items"] = self.num_unique_items
        config["k"] = self.k
        base_config = super(ItemCoverageAt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        num_unique_items = config.pop("num_unique_items", None)
        k = config.pop("k", None)

        return cls(
            num_unique_items=num_unique_items,
            k=k,
            **config,
        )
