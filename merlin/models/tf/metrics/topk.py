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
from typing import List, Optional, Sequence, Tuple, Union

import tensorflow as tf
from keras.utils import losses_utils, metrics_utils
from tensorflow.keras import backend
from tensorflow.keras.metrics import Mean, Metric
from tensorflow.keras.metrics import get as get_metric

from merlin.models.tf.metrics import metrics_registry
from merlin.models.tf.utils.tf_utils import extract_topk

METRIC_PARAMETERS_DOCSTRING = """
    y_true : tf.Tensor
        A tensor with shape (batch_size, n_items) corresponding to
        the multi-hot representation of true labels.
    y_pred : tf.Tensor
        A tensor with shape (batch_size, n_items) corresponding to
        the prediction scores.
    label_relevant_counts: tf.Tensor
        A 1D tensor (batch size) which contains the total number of relevant items
        for the example. This is necessary when setting `pre_sorting=True` on the
        ranking metrics classes (e.g. RecallAt(5, pre_sorted=True)), as extract_topk
        is used to extract only the top-k predictions and corresponding labels,
        potentially losing other relevant items not among top-k. In such cases,
        the label_relevant_counts will contain the total relevant counts per example.
    k : int
        The cut-off for ranking metrics
"""


def recall_at(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    label_relevant_counts: Optional[tf.Tensor] = None,
    k: int = 5,
) -> tf.Tensor:
    """
    Computes Recall@K metric
    ----------
    {METRIC_PARAMETERS_DOCSTRING}
    """
    rel_count = tf.clip_by_value(label_relevant_counts, clip_value_min=1, clip_value_max=float(k))

    rel_labels = tf.reduce_sum(y_true[:, :k], axis=-1)
    results = tf.cast(
        tf.math.divide_no_nan(rel_labels, rel_count),
        backend.floatx(),
    )
    return results


def precision_at(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    label_relevant_counts: Optional[tf.Tensor] = None,
    k: int = 5,
) -> tf.Tensor:
    """
    Computes Precision@K metric
    Parameters
    ----------
    {METRIC_PARAMETERS_DOCSTRING}
    """
    results = tf.cast(tf.reduce_mean(y_true[:, : int(k)], axis=-1), backend.floatx())
    return results


def average_precision_at(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    label_relevant_counts: Optional[tf.Tensor] = None,
    k: int = 5,
) -> tf.Tensor:
    """
    Computes Mean Average Precision (MAP) @K
    Parameters
    ----------
    {METRIC_PARAMETERS_DOCSTRING}
    """
    # Computing the precision from 1 to k range
    precisions = tf.stack([precision_at(y_true, y_pred, k=_k) for _k in range(1, k + 1)], axis=-1)
    # Keeping only the precision at the position of relevant items
    rel_precisions = precisions * y_true[:, :k]

    total_prec = tf.reduce_sum(rel_precisions, axis=-1)
    total_relevant_topk = tf.clip_by_value(
        label_relevant_counts, clip_value_min=1, clip_value_max=float(k)
    )

    results = tf.cast(tf.math.divide_no_nan(total_prec, total_relevant_topk), backend.floatx())

    return results


def dcg_at(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    label_relevant_counts: Optional[tf.Tensor] = None,
    k: int = 5,
    log_base: int = 2,
) -> tf.Tensor:
    """
    Compute discounted cumulative gain @K (ignoring ties)
    Parameters
    ----------
    {METRIC_PARAMETERS_DOCSTRING}
    """

    # Compute discounts
    discount_positions = tf.cast(tf.range(k), backend.floatx())
    discount_log_base = tf.math.log(tf.convert_to_tensor([log_base], dtype=backend.floatx()))

    discounts = 1 / (tf.math.log(discount_positions + 2) / discount_log_base)
    m = y_true[:, :k] * tf.repeat(tf.expand_dims(discounts[:k], 0), tf.shape(y_true)[0], axis=0)

    results = tf.cast(tf.reduce_sum(m, axis=-1), backend.floatx())
    return results


def ndcg_at(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    label_relevant_counts: Optional[tf.Tensor] = None,
    k: int = 5,
    log_base: int = 2,
) -> tf.Tensor:
    """
    Compute normalized discounted cumulative gain @K (ignoring ties)
    Parameters
    ----------
    {METRIC_PARAMETERS_DOCSTRING}
    log_base : int
        Base of the log discount where relevant items are rankied. Defaults to 2
    """
    gains = dcg_at(y_true, y_pred, k=k, log_base=log_base)
    perfect_labels_sorting = tf.cast(
        tf.cast(tf.expand_dims(tf.range(k), 0), label_relevant_counts.dtype)  # type: ignore
        < tf.expand_dims(label_relevant_counts, -1),
        backend.floatx(),
    )
    ideal_gains = dcg_at(perfect_labels_sorting, perfect_labels_sorting, k=k, log_base=log_base)

    results = tf.cast(tf.math.divide_no_nan(gains, ideal_gains), backend.floatx())
    return results


def mrr_at(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    label_relevant_counts: Optional[tf.Tensor] = None,
    k: int = 5,
) -> tf.Tensor:
    """
    Compute MRR
    ----------
    {METRIC_PARAMETERS_DOCSTRING}
    """

    first_rel_position = tf.cast(tf.argmax(y_true, axis=-1) + 1, backend.floatx())
    relevant_mask = tf.reduce_max(y_true[:, : int(k)], axis=-1)

    rel_position = first_rel_position * relevant_mask
    results = tf.cast(tf.math.divide_no_nan(1.0, rel_position), backend.floatx())

    return results


class TopkMetricWithLabelRelevantCountsMixin:
    @property
    def label_relevant_counts(self) -> tf.Tensor:
        return self._label_relevant_counts

    @label_relevant_counts.setter
    def label_relevant_counts(self, new_value: tf.Tensor):
        self._label_relevant_counts = new_value

    def _reshape_tensors(
        self,
        y_pred: tf.Tensor,
        y_true: tf.Tensor,
        label_relevant_counts: Optional[tf.Tensor],
        new_shape: Union[tf.TensorShape, tuple, list],
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        """Reshapes the predictions, targets and label_relevant_counts"""
        y_pred = tf.reshape(y_pred, new_shape)
        y_true = tf.reshape(y_true, new_shape)
        if label_relevant_counts is not None:
            label_relevant_counts = tf.reshape(label_relevant_counts, new_shape[:-1])
        return y_pred, y_true, label_relevant_counts


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TopkMetric(Mean, TopkMetricWithLabelRelevantCountsMixin):
    def __init__(self, fn, k=5, pre_sorted=True, name=None, log_base=None, seed=None, **kwargs):
        """Create instance of a TopKMetric.

        Parameters
        ----------
        fn : function
            Aggregtation function to use to compute metric passed y_true, y_pred
        k : int, optional
            Top k to compute metrics for, by default 5
        pre_sorted : bool, optional
            Whether or not the data passed to the metric is already sorted, by default True
        name : str, optional
            Name of the metric, by default None
        log_base : int, optional
            Base of the log discount where relevant items are ranked, by default None
        seed : int, optional
            Random seed to use for the shuffling in case of ties, by default None
        """
        self.name_orig = name
        if name is not None:
            name = f"{name}_{k}"
        super().__init__(name=name, **kwargs)
        self._fn = fn
        self.k = k
        self.seed = seed
        self._pre_sorted = pre_sorted
        self._fn_kwargs = {}
        if log_base is not None:
            self._fn_kwargs["log_base"] = log_base
        self.label_relevant_counts = None
        self.mask = None

    @property
    def pre_sorted(self):
        return self._pre_sorted

    @pre_sorted.setter
    def pre_sorted(self, new_value):
        self._pre_sorted = new_value

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ):
        y_true, y_pred = tf.squeeze(y_true), tf.squeeze(y_pred)
        tf.debugging.assert_greater_equal(
            tf.shape(y_true)[-1],
            self.k,
            f"The TopkMetric {self.name} cutoff ({self.k}) cannot be bigger than "
            f"the number of predictions per example",
        )

        y_true, y_pred = self.check_cast_inputs(y_true, y_pred)

        (
            [y_true, y_pred],
            sample_weight,
        ) = metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight
        )
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        label_relevant_counts = self.label_relevant_counts

        # For prediction tensor with rank > 2 (e.g. sequences)
        # reshapes the predictions, targets and label_relevant_counts
        # so that they are 2D and metrics work properly
        original_shape = tf.shape(y_pred)
        y_pred, y_true, label_relevant_counts = self._reshape_tensors(
            y_pred, y_true, label_relevant_counts, new_shape=[-1, original_shape[-1]]
        )

        y_pred, y_true, label_relevant_counts = self._maybe_sort_top_k(
            y_pred, y_true, label_relevant_counts
        )

        ag_fn = tf.__internal__.autograph.tf_convert(
            self._fn, tf.__internal__.autograph.control_status_ctx()
        )

        matches = ag_fn(
            y_true,
            y_pred,
            label_relevant_counts=label_relevant_counts,
            k=self.k,
            **self._fn_kwargs,
        )

        # Reshapes the metrics results so that they match the
        # original shape of predictions/targets before combining
        # with the sample weights
        matches = tf.reshape(matches, original_shape[:-1])

        return super().update_state(matches, sample_weight=sample_weight)

    def _maybe_sort_top_k(self, y_pred, y_true, label_relevant_counts: tf.Tensor = None):
        if not self.pre_sorted:
            y_pred, y_true, label_relevant_counts = extract_topk(
                self.k, y_pred, y_true, shuffle_ties=True, seed=self.seed
            )
        else:
            if label_relevant_counts is None:
                raise Exception(
                    "If y_true was pre-sorted (and truncated to top-k) you must "
                    "provide label_relevant_counts argument."
                )
            label_relevant_counts = tf.cast(label_relevant_counts, self._dtype)

        return y_pred, y_true, label_relevant_counts

    def check_cast_inputs(self, labels, predictions):
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        return tf.cast(labels, self._dtype), tf.cast(predictions, self._dtype)

    def get_config(self):
        config = {
            "name": self.name_orig,
            "k": self.k,
            "seed": self.seed,
            "pre_sorted": self._pre_sorted,
        }

        if type(self) is TopkMetric:
            # Only include function argument when the object is a TopkMetric and not a subclass.
            config["fn"] = self._fn

            for k, v in self._fn_kwargs.items():
                config[k] = backend.eval(v) if tf.is_tensor(v) or isinstance(v, tf.Variable) else v

        base_config = super(TopkMetric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if cls is TopkMetric:
            fn = config.pop("fn", None)
            k = config.pop("k", None)
            pre_sorted = config.pop("pre_sorted", None)
            seed = config.pop("seed", None)
            return cls(get_metric(fn), k=k, pre_sorted=pre_sorted, seed=seed, **config)
        return super(TopkMetric, cls).from_config(config)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
@metrics_registry.register_with_multiple_names("recall_at")
class RecallAt(TopkMetric):
    def __init__(self, k=10, pre_sorted=False, name="recall_at", **kwargs):
        super().__init__(recall_at, k=k, pre_sorted=pre_sorted, name=name, **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
@metrics_registry.register_with_multiple_names("precision_at")
class PrecisionAt(TopkMetric):
    def __init__(self, k=10, pre_sorted=False, name="precision_at", **kwargs):
        super().__init__(precision_at, k=k, pre_sorted=pre_sorted, name=name, **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
@metrics_registry.register_with_multiple_names("map_at", "map")
class AvgPrecisionAt(TopkMetric):
    def __init__(self, k=10, pre_sorted=False, name="map_at", **kwargs):
        super().__init__(average_precision_at, k=k, pre_sorted=pre_sorted, name=name, **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
@metrics_registry.register_with_multiple_names("mrr_at", "mrr")
class MRRAt(TopkMetric):
    def __init__(self, k=10, pre_sorted=False, name="mrr_at", **kwargs):
        super().__init__(mrr_at, k=k, pre_sorted=pre_sorted, name=name, **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
@metrics_registry.register_with_multiple_names("ndcg_at", "ndcg")
class NDCGAt(TopkMetric):
    def __init__(self, k=10, pre_sorted=False, name="ndcg_at", **kwargs):
        super().__init__(ndcg_at, k=k, pre_sorted=pre_sorted, name=name, **kwargs)


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class TopKMetricsAggregator(Metric, TopkMetricWithLabelRelevantCountsMixin):
    """Aggregator for top-k metrics (TopkMetric) that is optimized
    to sort top-k predictions only once for all metrics.

    *topk_metrics : TopkMetric
        Multiple arguments with TopkMetric instances
    """

    def __init__(self, *topk_metrics: TopkMetric, **kwargs):
        super(TopKMetricsAggregator, self).__init__(**kwargs)
        assert len(topk_metrics) > 0, "At least one topk_metrics should be provided"
        assert all(
            isinstance(m, TopkMetric) for m in topk_metrics
        ), "All provided metrics should inherit from TopkMetric"
        self.topk_metrics = topk_metrics

        # Setting the `pre_sorted` of topk metrics so that
        # prediction scores are not sorted again for each metric
        for metric in self.topk_metrics:
            metric.pre_sorted = True

        self.k = max([m.k for m in self.topk_metrics])

        self.label_relevant_counts = None

    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None
    ):
        # squeeze dim=1
        y_true, y_pred = tf.squeeze(y_true), tf.squeeze(y_pred)
        # For prediction tensor with rank > 2 (e.g. sequences)
        # reshapes the predictions, targets and label_relevant_counts
        # so that they are 2D and extract_topk() work properly
        original_shape = tf.shape(y_pred)
        y_pred, y_true, _ = self._reshape_tensors(
            y_pred, y_true, None, new_shape=[-1, original_shape[-1]]
        )

        # Extracting sorted top-k prediction scores and labels only ONCE
        # so that sorting does not need to happen for each individual metric
        # (as the top-k metrics have been set with pre_sorted=True in this constructor
        y_pred, y_true, label_relevant_counts_from_targets = extract_topk(
            self.k, y_pred, y_true, shuffle_ties=True
        )

        # Reshaping tensors back to their original shape (expect for the last dim that
        # equals to k)
        y_pred, y_true, label_relevant_counts_from_targets = self._reshape_tensors(
            y_pred,
            y_true,
            label_relevant_counts_from_targets,
            new_shape=tf.concat([original_shape[:-1], [-1]], axis=-1),
        )

        # If label_relevant_counts is not set by a block (e.g. TopKIndexBlock) that
        # has already extracted the top-k predictions, it is assumed that
        # y_true contains all items
        label_relevant_counts = self.label_relevant_counts
        if label_relevant_counts is None:
            label_relevant_counts = label_relevant_counts_from_targets

        for metric in self.topk_metrics:
            # Sets the label_relevant_counts using a property,
            # as metric.update_state() should have standard signature
            # for better compatibility with Keras
            metric.label_relevant_counts = label_relevant_counts
            metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        outputs = {}
        weighted = "weighted_" in self._name
        for metric in self.topk_metrics:
            name = metric.name
            if weighted:
                name = "weighted_" + name
            outputs[name] = metric.result()

        return outputs

    @classmethod
    def default_metrics(cls, top_ks: Sequence[int], **kwargs) -> Sequence[TopkMetric]:
        """Returns an TopKMetricsAggregator instance with the default top-k metrics
        at the cut-offs defined in top_ks

        Parameters
        ----------
        top_ks : Sequence[int]
            List with the cut-offs for top-k metrics (e.g. [5,10,50])

        Returns
        -------
        Sequence[TopkMetric]
            A TopKMetricsAggregator instance with the default top-k metrics at the predefined
            cut-offs
        """
        metrics: List[TopkMetric] = []
        for k in top_ks:
            metrics.extend([RecallAt(k), MRRAt(k), NDCGAt(k), AvgPrecisionAt(k), PrecisionAt(k)])
        # Using Top-k metrics aggregator provides better performance than having top-k
        # metrics computed separately, as prediction scores are sorted only once for all metrics
        aggregator = cls(*metrics)
        return [aggregator]

    def get_config(self):
        config = super(TopKMetricsAggregator, self).get_config()
        config["topk_metrics"] = [tf.keras.layers.serialize(metric) for metric in self.topk_metrics]
        return config

    @classmethod
    def from_config(cls, config):
        topk_metrics = config.pop("topk_metrics")
        topk_metrics = [
            tf.keras.layers.deserialize(metric_config) for metric_config in topk_metrics
        ]
        return cls(*topk_metrics, **config)


def filter_topk_metrics(
    metrics: Sequence[Metric],
) -> List[Union[TopkMetric, TopKMetricsAggregator]]:
    """Returns only top-k metrics from the list of metrics

    Parameters
    ----------
    metrics : List[Metric]
        List of metrics

    Returns
    -------
    List[Union[TopkMetric, TopKMetricsAggregator]]
        List with the top-k metrics in the list of input metrics
    """
    topk_metrics = list(
        [
            metric
            for metric in metrics
            if isinstance(metric, TopkMetric) or isinstance(metric, TopKMetricsAggregator)
        ]
    )
    return topk_metrics


def split_metrics(
    metrics: Sequence[Metric],
    return_other_metrics: bool = False,
) -> Tuple[TopkMetric, TopKMetricsAggregator, Metric]:
    """Split the list of metrics into top-k metrics, top-k aggregators and others

    Parameters
    ----------
    metrics : List[Metric]
        List of metrics

    Returns
    -------
    List[TopkMetric, TopKMetricsAggregator, Metric]
        List with the top-k metrics in the list of input metrics
    """
    topk_metrics, topk_aggregators, other_metrics = [], [], []
    for metric in metrics:
        if isinstance(metric, str):
            metric = metrics_registry.parse(metric)

        if isinstance(metric, TopkMetric):
            topk_metrics.append(metric)
        elif isinstance(metric, TopKMetricsAggregator):
            topk_aggregators.append(metric)
        else:
            other_metrics.append(metric)
    return topk_metrics, topk_aggregators, other_metrics
