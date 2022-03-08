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

import pytest
import tensorflow as tf

from merlin.models.tf.metrics.ranking import RecallAt2, recall_at
from merlin.models.tf.utils.tf_utils import extract_topk


@pytest.fixture
def ranking_metrics_test_data():
    labels = tf.convert_to_tensor([[0, 1, 0, 0, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], tf.float32)
    predictions = tf.convert_to_tensor(
        [[10, 9, 8, 7, 6], [10, 9, 8, 7, 6], [1, 4, 3, 2, 5]], tf.float32
    )
    label_relevant_counts = tf.convert_to_tensor([2, 0, 1], tf.float32)
    return labels, predictions, label_relevant_counts


@pytest.mark.parametrize(
    "metric_exp_result",
    [(RecallAt2(k=3, pre_sorted=False), 0.25)],
)
def test_ranking_metrics_classes_not_pre_sorted(ranking_metrics_test_data, metric_exp_result):
    labels, predictions, _ = ranking_metrics_test_data
    metric, expected_result = metric_exp_result

    metric.update_state(labels, predictions)

    result = metric.result()
    tf.assert_equal(result, expected_result)


@pytest.mark.parametrize(
    "metric_exp_result",
    [(RecallAt2(k=3, pre_sorted=True), 0.25)],
)
def test_ranking_metrics_classes_pre_sorted(ranking_metrics_test_data, metric_exp_result):
    metric, expected_result = metric_exp_result

    labels, predictions, label_relevant_counts = ranking_metrics_test_data
    # Pre-sorting predictions and labels by predictions
    predictions, labels, label_relevant_counts = extract_topk(5, predictions, labels)

    metric.update_state(labels, predictions, label_relevant_counts=label_relevant_counts)

    result = metric.result()
    tf.assert_equal(result, expected_result)


def test_ranking_metrics_not_pre_sorted(ranking_metrics_test_data):
    labels, predictions, _ = ranking_metrics_test_data

    metric = RecallAt2(k=3, pre_sorted=True)

    with pytest.raises(Exception) as excinfo:
        metric.update_state(labels, predictions)
    assert "you must provide label_relevant_counts argument" in str(excinfo.value)


@pytest.mark.parametrize("metric_exp_result", [(recall_at, 0.25)])
def test_ranking_metrics_pre_sorted(ranking_metrics_test_data, metric_exp_result):
    metric, expected_result = metric_exp_result

    # Pre-sorting predictions and labels by predictions
    labels, predictions, label_relevant_counts = ranking_metrics_test_data
    predictions, labels, label_relevant_counts = extract_topk(5, predictions, labels)

    result = metric(labels, predictions, label_relevant_counts, k=3)

    tf.assert_equal(tf.reduce_mean(result), expected_result)
