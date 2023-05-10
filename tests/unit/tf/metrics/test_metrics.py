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
from tensorflow.keras.losses import binary_crossentropy

from merlin.models.tf.metrics import metrics_registry
from merlin.models.tf.metrics.evaluation import (
    ItemCoverageAt,
    LogLossMetric,
    NoveltyAt,
    PopularityBiasAt,
)


@pytest.mark.parametrize("metric", [LogLossMetric(), "logloss"])
def test_logloss_metric(metric):
    metric = metrics_registry.parse(metric)

    y_pred = tf.convert_to_tensor([1.0, 0.5, 0.2, 2.3, 5.0], tf.float32)
    y_true = tf.convert_to_tensor([0, 1, 0, 0, 1], tf.int32)

    expected_result = binary_crossentropy(y_true, y_pred, from_logits=False)

    metric.update_state(y_true, y_pred)
    result = metric.result()

    tf.debugging.assert_near(result, expected_result)


@pytest.fixture
def popularity_metrics_test_data():
    labels = tf.convert_to_tensor([[1], [3], [7], [6]], tf.int64)
    predictions = tf.convert_to_tensor(
        [[1, 9, 8, 7, 6], [1, 4, 3, 2, 5], [5, 9, 8, 7, 6]], tf.int64
    )
    item_freq_probs = [0.1, 0.05, 0.2, 0.05, 0.1, 0.1, 0.15, 0.05, 0.1, 0.1]
    return labels, predictions, item_freq_probs


@pytest.mark.parametrize("is_prob_distribution", [True, False])
def test_novelty_at_k(popularity_metrics_test_data, is_prob_distribution):
    labels, predictions, item_freq_probs = popularity_metrics_test_data
    metric1 = NoveltyAt(
        item_freq_probs=item_freq_probs, is_prob_distribution=is_prob_distribution, k=3
    )

    metric1.update_state(predicted_ids=predictions)
    result1 = metric1.result()
    tf.debugging.assert_near(result1, 2.5336342)


@pytest.mark.parametrize("is_prob_distribution", [True, False])
def test_popularity_at_k(popularity_metrics_test_data, is_prob_distribution):
    labels, predictions, item_freq_probs = popularity_metrics_test_data

    metric1 = PopularityBiasAt(
        item_freq_probs=item_freq_probs, is_prob_distribution=is_prob_distribution, k=3
    )

    metric1.update_state(predicted_ids=predictions)
    result1 = metric1.result()
    tf.debugging.assert_near(result1, 0.0833333)


def test_item_coverage_at_k(popularity_metrics_test_data):
    labels, predictions, item_freq_probs = popularity_metrics_test_data

    metric1 = ItemCoverageAt(num_unique_items=11, k=4)

    metric1.update_state(predicted_ids=predictions)
    result1 = metric1.result()
    tf.debugging.assert_near(result1, 0.727272)
