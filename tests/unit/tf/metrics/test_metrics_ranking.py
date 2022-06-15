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

from merlin.models.tf.metrics.ranking import (
    AvgPrecisionAt,
    MRRAt,
    NDCGAt,
    PrecisionAt,
    RecallAt,
    average_precision_at,
    dcg_at,
    mrr_at,
    ndcg_at,
    precision_at,
    recall_at,
)
from merlin.models.tf.utils.tf_utils import extract_topk


@pytest.fixture
def ranking_metrics_test_data():
    labels = tf.convert_to_tensor([[0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 1]], tf.float32)
    predictions = tf.convert_to_tensor(
        [[10, 9, 8, 7, 6], [1, 4, 3, 2, 5], [10, 9, 8, 7, 6]], tf.float32
    )
    label_relevant_counts = tf.convert_to_tensor([2, 2, 0], tf.float32)
    return labels, predictions, label_relevant_counts


@pytest.fixture
def ranking_metrics_test_data_pre_sorted(ranking_metrics_test_data):
    labels, predictions, label_relevant_counts = ranking_metrics_test_data
    predictions, labels, label_relevant_counts = extract_topk(5, predictions, labels)
    return labels, predictions, label_relevant_counts


def test_ranking_metrics_not_pre_sorted(ranking_metrics_test_data):
    labels, predictions, _ = ranking_metrics_test_data

    metric = RecallAt(k=4, pre_sorted=True)

    with pytest.raises(Exception) as excinfo:
        metric.update_state(labels, predictions)
    assert "you must provide label_relevant_counts argument" in str(excinfo.value)


def test_recall_at_k(ranking_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = ranking_metrics_test_data_pre_sorted
    result = recall_at(labels, predictions, label_relevant_counts, k=4)

    expected_result = [2 / 2, 1 / 2, 0]  # The last example has 0 relevant items
    tf.assert_equal(result, expected_result)


def test_precision_at_k(ranking_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = ranking_metrics_test_data_pre_sorted
    k = 4
    result = precision_at(labels, predictions, label_relevant_counts, k=k)

    expected_result = [2 / k, 1 / k, 0 / k]
    tf.assert_equal(result, expected_result)


def test_average_precision_at(ranking_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = ranking_metrics_test_data_pre_sorted
    result = average_precision_at(labels, predictions, label_relevant_counts, k=4)

    # Averaged precision at the position of relevant items among top-k,
    # divided by the total number of relevant items
    expected_result = [(1 / 2 + 2 / 4) / 2, (1 / 4) / 2, 0]  # The last example has 0 relevant items
    tf.assert_equal(result, expected_result)


def dcg_probe(pos_relevant, relevant_score=1):
    return relevant_score / math.log(pos_relevant + 1, 2)


def test_dcg_at(ranking_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = ranking_metrics_test_data_pre_sorted
    result = dcg_at(labels, predictions, label_relevant_counts, k=4)
    expected_result = [
        dcg_probe(2) + dcg_probe(4),
        dcg_probe(4),
        0,
    ]  # The last example has 0 relevant items
    tf.debugging.assert_near(result, expected_result)


def test_ndcg_at(ranking_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = ranking_metrics_test_data_pre_sorted
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


def test_mrr_at(ranking_metrics_test_data_pre_sorted):
    labels, predictions, label_relevant_counts = ranking_metrics_test_data_pre_sorted
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
def test_ranking_metrics_classes(ranking_metrics_test_data_pre_sorted, metric_exp_result):
    labels, predictions, label_relevant_counts = ranking_metrics_test_data_pre_sorted
    metric, expected_result = metric_exp_result

    metric.update_state(labels, predictions, label_relevant_counts)

    result = metric.result()
    tf.debugging.assert_near(result, expected_result)


@pytest.mark.parametrize(
    "metric_class",
    [
        RecallAt,
        PrecisionAt,
        AvgPrecisionAt,
        MRRAt,
        NDCGAt,
    ],
)
def test_ranking_metrics_classes_pre_or_not_sorted_matches(ranking_metrics_test_data, metric_class):
    labels, predictions, label_relevant_counts = ranking_metrics_test_data
    # Pre-sorting predictions and labels by predictions
    predictions_sorted, labels_sorted, label_relevant_counts_sorted = extract_topk(
        4, predictions, labels
    )

    metric1 = metric_class(k=4, pre_sorted=True)
    metric1.update_state(
        labels_sorted, predictions_sorted, label_relevant_counts=label_relevant_counts_sorted
    )
    result1 = metric1.result()

    metric2 = metric_class(k=4, pre_sorted=False)
    metric2.update_state(labels, predictions, label_relevant_counts=label_relevant_counts)
    result2 = metric2.result()

    tf.assert_equal(result1, result2)
