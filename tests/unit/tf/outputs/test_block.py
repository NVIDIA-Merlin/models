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
from tensorflow.keras.metrics import Metric

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


@testing_utils.mark_run_eagerly_modes
def test_model_output(ecommerce_data: Dataset, run_eagerly: bool):
    model = mm.Model(
        mm.InputBlockV2(ecommerce_data.schema),
        mm.MLPBlock([4]),
        mm.OutputBlock(ecommerce_data.schema),
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "task_blocks",
    [
        None,
        mm.MLPBlock([32]),
        dict(click=mm.MLPBlock([16]), play_percentage=mm.MLPBlock([20])),
        {
            "click/binary_output": mm.MLPBlock([16]),
            "play_percentage/regression_output": mm.MLPBlock([20]),
        },
    ],
)
def test_model_with_multi_output_blocks_with_task_towers(
    music_streaming_data: Dataset, task_blocks, run_eagerly: bool
):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(music_streaming_data.schema, task_blocks=task_blocks)
    model = mm.Model(inputs, mm.MLPBlock([64]), output_block)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(mm.sample_batch(music_streaming_data, batch_size=50))

    assert metrics["loss"] >= 0
    assert set(list(metrics.keys())) == set(
        [
            "loss",
            "loss_batch",
            "regularization_loss",
            "click/binary_output_loss",
            "like/binary_output_loss",
            "play_percentage/regression_output_loss",
            "click/binary_output/precision",
            "click/binary_output/recall",
            "click/binary_output/binary_accuracy",
            "click/binary_output/auc",
            "like/binary_output/precision",
            "like/binary_output/recall",
            "like/binary_output/binary_accuracy",
            "like/binary_output/auc",
            "play_percentage/regression_output/root_mean_squared_error",
        ]
    )
    if task_blocks:
        # Checking that task blocks (first layer from SequenceBlock) are different for every task
        assert (
            output_block.parallel_dict["click/binary_output"][0]
            != output_block.parallel_dict["play_percentage/regression_output"][0]
        )
        if isinstance(task_blocks, dict):
            # Ensures for like there is no task tower
            assert isinstance(output_block.parallel_dict["like/binary_output"], mm.BinaryOutput)


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "metrics",
    [
        None,
        tf.keras.metrics.AUC(name="auc"),
        (tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")),
        {
            "click/binary_output": (
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ),
            "like/binary_output": tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        },
    ],
)
def test_model_with_multi_output_blocks_metrics_tasks(
    music_streaming_data: Dataset, run_eagerly: bool, metrics
):
    music_streaming_data.schema = music_streaming_data.schema.without("play_percentage")

    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(music_streaming_data.schema)
    model = mm.Model(inputs, mm.MLPBlock([64]), output_block)

    weighted_metrics = metrics

    expected_metrics = [
        "loss",
        "regularization_loss",
        "loss_batch",
        "click/binary_output_loss",
        "like/binary_output_loss",
    ]
    if isinstance(metrics, Metric):
        expected_metrics.extend(
            [
                "click/binary_output/auc",
                "like/binary_output/auc",
                "click/binary_output/weighted_auc",
                "like/binary_output/weighted_auc",
            ]
        )
    elif isinstance(metrics, (list, tuple)):
        expected_metrics.extend(
            [
                "click/binary_output/precision",
                "like/binary_output/precision",
                "click/binary_output/recall",
                "like/binary_output/recall",
                "click/binary_output/weighted_precision",
                "like/binary_output/weighted_precision",
                "click/binary_output/weighted_recall",
                "like/binary_output/weighted_recall",
            ]
        )
    elif isinstance(metrics, dict):
        expected_metrics.extend(
            [
                "click/binary_output/precision",
                "click/binary_output/recall",
                "like/binary_output/binary_accuracy",
                "click/binary_output/weighted_precision",
                "click/binary_output/weighted_recall",
                "like/binary_output/weighted_binary_accuracy",
            ]
        )
        # Creating another copy of the metrics to avoid error for reusing same metrics
        # for metrics and weighted_metrics
        weighted_metrics = {
            "click/binary_output": (
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ),
            "like/binary_output": tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        }
    elif metrics is None:
        # Use default metrics
        expected_metrics.extend(
            [
                "click/binary_output/precision",
                "click/binary_output/recall",
                "click/binary_output/binary_accuracy",
                "click/binary_output/auc",
                "like/binary_output/precision",
                "like/binary_output/recall",
                "like/binary_output/binary_accuracy",
                "like/binary_output/auc",
            ]
        )

    model.compile(
        optimizer="adam",
        run_eagerly=run_eagerly,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )

    metrics_results = model.train_step(mm.sample_batch(music_streaming_data, batch_size=50))

    assert metrics_results["loss"] >= 0
    assert set(metrics_results.keys()) == set(expected_metrics)


@testing_utils.mark_run_eagerly_modes
def test_model_with_multi_output_blocks_loss_weights_and_weighted_metrics(
    music_streaming_data: Dataset, run_eagerly: bool
):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(music_streaming_data.schema)
    model = mm.Model(inputs, mm.MLPBlock([64]), output_block)

    loss_weights = {
        "click/binary_output": 1.0,
        "play_percentage/regression_output": 2.0,
        "like/binary_output": 3.0,
    }

    weighted_metrics = {
        "click/binary_output": (
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ),
        "like/binary_output": (
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ),
        "play_percentage/regression_output": (
            tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
        ),
    }

    model.compile(
        optimizer="adam",
        run_eagerly=run_eagerly,
        loss_weights=loss_weights,
        weighted_metrics=weighted_metrics,
    )

    batch = mm.sample_batch(music_streaming_data, batch_size=50)

    metrics = model.test_step(batch)

    assert metrics["loss"] >= 0
    assert set(metrics.keys()) == set(
        [
            "loss",
            "click/binary_output_loss",
            "like/binary_output_loss",
            "play_percentage/regression_output_loss",
            "click/binary_output/precision",
            "click/binary_output/recall",
            "click/binary_output/binary_accuracy",
            "click/binary_output/auc",
            "click/binary_output/weighted_precision",
            "click/binary_output/weighted_recall",
            "click/binary_output/weighted_binary_accuracy",
            "click/binary_output/weighted_auc",
            "like/binary_output/precision",
            "like/binary_output/recall",
            "like/binary_output/binary_accuracy",
            "like/binary_output/auc",
            "like/binary_output/weighted_precision",
            "like/binary_output/weighted_recall",
            "like/binary_output/weighted_binary_accuracy",
            "like/binary_output/weighted_auc",
            "play_percentage/regression_output/root_mean_squared_error",
            "play_percentage/regression_output/weighted_root_mean_squared_error",
            "regularization_loss",
            "loss_batch",
        ]
    )

    metrics2 = model.test_step(batch)
    assert metrics["loss"] == metrics2["loss"]
    for m in metrics:
        assert metrics[m] == metrics2[m]

    # Disabling losses weights
    model.compile(
        loss_weights=None,
    )
    metrics_non_weighted = model.test_step(batch)

    assert set(metrics_non_weighted.keys()) == set(list(metrics.keys())).difference(
        set(
            [
                "click/binary_output/weighted_precision",
                "click/binary_output/weighted_recall",
                "click/binary_output/weighted_binary_accuracy",
                "click/binary_output/weighted_auc",
                "like/binary_output/weighted_precision",
                "like/binary_output/weighted_recall",
                "like/binary_output/weighted_binary_accuracy",
                "like/binary_output/weighted_auc",
                "play_percentage/regression_output/weighted_root_mean_squared_error",
            ]
        )
    )

    for m in metrics_non_weighted:
        if m in ["loss", "loss_batch"]:
            assert metrics[m] != metrics_non_weighted[m]
        else:
            assert metrics[m] == metrics_non_weighted[m]
