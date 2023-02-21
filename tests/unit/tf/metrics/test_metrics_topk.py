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

import math

import pytest
import tensorflow as tf
from sklearn.metrics import ndcg_score as ndcg_score_sklearn

from merlin.models.tf.metrics.topk import (
    AvgPrecisionAt,
    MRRAt,
    NDCGAt,
    PrecisionAt,
    RecallAt,
    TopKMetricsAggregator,
    average_precision_at,
    dcg_at,
    mrr_at,
    ndcg_at,
    precision_at,
    recall_at,
)
from merlin.models.tf.utils.tf_utils import extract_topk


def test_ndcg_with_ties_seed():
    y_true = tf.constant([1, 1, 1, 2])
    y_pred = tf.constant([1, 2, 1, 2])
    results = []
    for _ in range(10):
        metric = NDCGAt(4, seed=44)
        metric.update_state(y_true, y_pred)
        results.append(metric.result().numpy())
    assert len(set(results)) == 1


@pytest.fixture
def topk_metrics_test_data():
    labels = tf.convert_to_tensor([[0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 1]], tf.float32)
    predictions = tf.convert_to_tensor(
        [[10, 9, 8, 7, 6], [1, 4, 3, 2, 5], [10, 9, 8, 7, 6]], tf.float32
    )
    label_relevant_counts = tf.convert_to_tensor([2, 2, 1], tf.float32)
    return labels, predictions, label_relevant_counts


@pytest.fixture
def topk_metrics_test_data_pre_sorted(topk_metrics_test_data):
    labels, predictions, _ = topk_metrics_test_data
    predictions, labels, label_relevant_counts = extract_topk(
        5, predictions, labels, shuffle_ties=True
    )
    return labels, predictions, label_relevant_counts


def test_topk_metrics_pre_sorted(topk_metrics_test_data_pre_sorted):
    labels, predictions, _ = topk_metrics_test_data_pre_sorted

    metric = RecallAt(k=4, pre_sorted=True)

    with pytest.raises(Exception) as excinfo:
        metric.update_state(labels, predictions)
    assert "you must provide label_relevant_counts argument" in str(excinfo.value)


def test_recall_at_k(topk_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = topk_metrics_test_data_pre_sorted
    result = recall_at(labels, predictions, label_relevant_counts, k=4)

    expected_result = [2 / 2, 1 / 2, 0 / 1]  # The last example has 0 relevant items
    tf.assert_equal(result, expected_result)


def test_precision_at_k(topk_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = topk_metrics_test_data_pre_sorted
    k = 4
    result = precision_at(labels, predictions, label_relevant_counts, k=k)

    expected_result = [2 / k, 1 / k, 0 / k]
    tf.assert_equal(result, expected_result)


def test_average_precision_at(topk_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = topk_metrics_test_data_pre_sorted
    result = average_precision_at(labels, predictions, label_relevant_counts, k=4)

    # Averaged precision at the position of relevant items among top-k,
    # divided by the total number of relevant items
    expected_result = [(1 / 2 + 2 / 4) / 2, (1 / 4) / 2, 0]  # The last example has 0 relevant items
    tf.assert_equal(result, expected_result)


def dcg_probe(pos_relevant, relevant_score=1):
    return relevant_score / math.log(pos_relevant + 1, 2)


def test_dcg_at(topk_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = topk_metrics_test_data_pre_sorted
    result = dcg_at(labels, predictions, label_relevant_counts, k=4)
    expected_result = [
        dcg_probe(2) + dcg_probe(4),
        dcg_probe(4),
        0,
    ]  # The last example has 0 relevant items
    tf.debugging.assert_near(result, expected_result)


def test_ndcg_at(topk_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = topk_metrics_test_data_pre_sorted
    result = ndcg_at(labels, predictions, label_relevant_counts, k=4)
    expected_result = [
        (dcg_probe(2) + dcg_probe(4)) / (dcg_probe(1) + dcg_probe(2)),
        dcg_probe(4) / (dcg_probe(1) + dcg_probe(2)),
        0,
    ]  # The last example has 0 relevant items
    tf.debugging.assert_near(result, expected_result)

    # Comparing with scikit learn
    result_sklearn = ndcg_score_sklearn(labels.numpy(), predictions.numpy(), k=4, ignore_ties=True)
    tf.debugging.assert_near(tf.cast(result_sklearn, tf.float32), tf.reduce_mean(expected_result))


def test_mrr_at(topk_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = topk_metrics_test_data_pre_sorted
    result = mrr_at(labels, predictions, label_relevant_counts, k=4)
    expected_result = [1 / 2, 1 / 4, 0]  # The last example has 0 relevant items
    tf.debugging.assert_near(result, expected_result)


@pytest.mark.parametrize(
    "metric_exp_result",
    [
        (RecallAt(k=4), 0.5),
        (PrecisionAt(k=4), 0.25),
        (AvgPrecisionAt(k=4), 0.20833333),
        (MRRAt(k=4), 0.25),
        (NDCGAt(k=4), 0.30499637),
    ],
)
def test_topk_metrics_classes(topk_metrics_test_data_pre_sorted, metric_exp_result):
    labels, predictions, label_relevant_counts = topk_metrics_test_data_pre_sorted
    metric, expected_result = metric_exp_result
    metric.label_relevant_counts = label_relevant_counts
    metric.update_state(labels, predictions)
    result = metric.result()
    tf.debugging.assert_near(result, expected_result)


def test_topk_metrics_aggregator(topk_metrics_test_data):
    metric_exp_result = [
        ("recall_at_4", RecallAt(k=4), 0.5),
        ("precision_at_4", PrecisionAt(k=4), 0.25),
        ("map_at_4", AvgPrecisionAt(k=4), 0.20833333),
        ("mrr_at_4", MRRAt(k=4), 0.25),
        ("ndcg_at_4", NDCGAt(k=4), 0.30499637),
    ]

    metric_names, metrics, expected_results = zip(*metric_exp_result)
    labels, predictions, label_relevant_counts = topk_metrics_test_data
    topk_metrics_aggregator = TopKMetricsAggregator(*metrics)
    topk_metrics_aggregator.label_relevant_counts = label_relevant_counts
    topk_metrics_aggregator.update_state(labels, predictions, None)

    results = topk_metrics_aggregator.result()
    assert set(results.keys()) == set(metric_names)
    for metric_name, exp_result in zip(metric_names, expected_results):
        assert results[metric_name] == exp_result


@pytest.mark.parametrize(
    "metric_class",
    [RecallAt, PrecisionAt, AvgPrecisionAt, MRRAt, NDCGAt],
)
def test_topk_metrics_classes_pre_or_not_sorted_matches(
    topk_metrics_test_data, topk_metrics_test_data_pre_sorted, metric_class
):
    labels, predictions, label_relevant_counts = topk_metrics_test_data
    # Pre-sorting predictions and labels
    predictions_sorted, labels_sorted, label_relevant_counts_sorted = extract_topk(
        4, predictions, labels, shuffle_ties=True
    )

    metric1 = metric_class(k=4, pre_sorted=True)
    metric1.label_relevant_counts = label_relevant_counts_sorted
    metric1.update_state(labels_sorted, predictions_sorted)
    result1 = metric1.result()

    metric2 = metric_class(k=4, pre_sorted=False)
    metric2.label_relevant_counts = label_relevant_counts
    metric2.update_state(labels, predictions)
    result2 = metric2.result()

    tf.assert_equal(result1, result2)


@pytest.mark.parametrize(
    "metric_class",
    [RecallAt, PrecisionAt, AvgPrecisionAt, MRRAt, NDCGAt],
)
def test_topk_reload(topk_metrics_test_data, metric_class):
    labels, predictions, label_relevant_counts = topk_metrics_test_data

    metric = metric_class(k=3, pre_sorted=False)
    metric.label_relevant_counts = label_relevant_counts
    metric.update_state(labels, predictions)
    result = metric.result()

    serialized = tf.keras.layers.serialize(metric)

    reloaded_metric = tf.keras.layers.deserialize(serialized)
    reloaded_metric.label_relevant_counts = label_relevant_counts
    reloaded_metric.update_state(labels, predictions)
    reloaded_result = reloaded_metric.result()

    tf.assert_equal(result, reloaded_result)


def test_topk_metrics_sequential_3d_with_sample_weights():
    labels = tf.convert_to_tensor(
        [
            [[0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]],
            [[0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]],
        ],
        tf.float32,
    )

    predictions = tf.convert_to_tensor(
        [
            [[10, 9, 8, 7, 6], [5, 4, 3, 2, 1], [10, 9, 8, 7, 6]],
            [[10, 9, 8, 7, 6], [5, 4, 3, 2, 1], [10, 9, 8, 7, 6]],
        ],
        tf.float32,
    )
    label_relevant_counts = tf.convert_to_tensor([[2, 2, 1], [2, 2, 1]], tf.float32)

    metric = RecallAt(k=4)
    metric.label_relevant_counts = label_relevant_counts

    # No masking
    metric.update_state(labels, predictions)
    result = metric.result()
    tf.debugging.assert_near(result, 0.5)

    # Using sample weights to mask some positions
    sample_weight = tf.convert_to_tensor([[0, 1, 0], [1, 0, 0]], tf.float32)
    metric.reset_state()
    metric.update_state(labels, predictions, sample_weight=sample_weight)
    result = metric.result()
    tf.debugging.assert_near(result, (1.0 + 0.5) / 2)
