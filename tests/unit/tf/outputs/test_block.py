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

import pytest
import tensorflow as tf
from tensorflow.keras.metrics import Metric

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.core.base import Block
from merlin.models.tf.utils import testing_utils


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize("use_output_block", [True, False])
def test_model_output(ecommerce_data: Dataset, run_eagerly: bool, use_output_block: bool):
    if use_output_block:
        output_block = mm.OutputBlock(ecommerce_data.schema)
    else:
        output_block = mm.ParallelBlock(
            mm.BinaryOutput("click"), mm.BinaryOutput("conversion", logits_temperature=2.0)
        )

    model = mm.Model(
        mm.InputBlockV2(ecommerce_data.schema),
        mm.MLPBlock([4]),
        output_block,
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "loss_batch",
        "click/binary_output_loss",
        "click/binary_output/precision",
        "click/binary_output/recall",
        "click/binary_output/binary_accuracy",
        "click/binary_output/auc",
        "conversion/binary_output_loss",
        "conversion/binary_output/precision",
        "conversion/binary_output/recall",
        "conversion/binary_output/binary_accuracy",
        "conversion/binary_output/auc",
        "regularization_loss",
    }


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "custom_task_output",
    [False, True],
)
def test_model_output_custom_task_output(
    ecommerce_data: Dataset, run_eagerly: bool, custom_task_output
):
    output_block = mm.OutputBlock(
        ecommerce_data.schema,
        model_outputs={"click/regression_output": mm.RegressionOutput("click")}
        if custom_task_output
        else None,
    )

    assert isinstance(output_block.parallel_dict["click/binary_output"], mm.BinaryOutput)
    assert isinstance(output_block.parallel_dict["conversion/binary_output"], mm.BinaryOutput)
    if custom_task_output:
        assert isinstance(
            output_block.parallel_dict["click/regression_output"], mm.RegressionOutput
        )

    model = mm.Model(mm.InputBlockV2(ecommerce_data.schema), mm.MLPBlock([4]), output_block)

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(mm.sample_batch(ecommerce_data, batch_size=50))

    expected_metrics = [
        "loss",
        "click/binary_output_loss",
        "conversion/binary_output_loss",
        "click/binary_output/precision",
        "click/binary_output/recall",
        "click/binary_output/binary_accuracy",
        "click/binary_output/auc",
        "conversion/binary_output/precision",
        "conversion/binary_output/recall",
        "conversion/binary_output/binary_accuracy",
        "conversion/binary_output/auc",
        "regularization_loss",
        "loss_batch",
    ]

    if custom_task_output:
        expected_metrics.extend(
            ["click/regression_output_loss", "click/regression_output/root_mean_squared_error"]
        )

    assert set(list(metrics.keys())) == set(expected_metrics)


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

    metrics = model.train_step(
        mm.sample_batch(music_streaming_data, batch_size=50, prepare_features=False)
    )

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
    assert (output_block.parallel_dict["click/binary_output"].pre is not None) == (
        task_blocks is not None
    )
    assert (output_block.parallel_dict["play_percentage/regression_output"].pre is not None) == (
        task_blocks is not None
    )
    if task_blocks:
        # Checking that task blocks (first layer from SequenceBlock) are different for every task
        assert (
            output_block.parallel_dict["click/binary_output"].pre
            != output_block.parallel_dict["play_percentage/regression_output"].pre
        )
        if isinstance(task_blocks, dict):
            # Ensures for like there is no task tower
            assert isinstance(output_block.parallel_dict["like/binary_output"], mm.BinaryOutput)


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        "auc",
        tf.keras.metrics.AUC(name="auc"),
        ("precision", "recall"),
        (tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")),
        (("precision", "recall"), ("binary_accuracy")),
        (
            (tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        ),
        {"click/binary_output": ("precision", "recall"), "like/binary_output": "binary_accuracy"},
        {
            "click/binary_output": (
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ),
            "like/binary_output": tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        },
    ],
)
def test_model_with_multi_output_blocks_metrics_tasks(music_streaming_data: Dataset, metrics):
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
    if isinstance(metrics, (str, Metric)):
        expected_metrics.extend(
            [
                "click/binary_output/auc",
                "like/binary_output/auc",
                "click/binary_output/weighted_auc",
                "like/binary_output/weighted_auc",
            ]
        )
    elif isinstance(metrics, (list, tuple)) and isinstance(metrics[0], (list, tuple)):
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
        weighted_metrics = (
            (tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")),
            (tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")),
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
        run_eagerly=True,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )

    metrics_results = model.train_step(
        mm.sample_batch(music_streaming_data, batch_size=50, prepare_features=False)
    )

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


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "sample_weight_column",
    ["like", "click", "user_age"],
)
def test_column_based_sample_weight(
    music_streaming_data: Dataset, sample_weight_column: str, run_eagerly: bool
):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.BinaryOutput(
        "like",
        post=mm.ColumnBasedSampleWeight(
            weight_column_name=sample_weight_column,
            binary_class_weights=((1.0, 5.0) if sample_weight_column == "like" else None),
        ),
    )

    model = mm.Model(inputs, mm.MLPBlock([8]), output_block)

    model.compile(
        optimizer="adam",
        run_eagerly=run_eagerly,
        weighted_metrics=["auc"],
    )

    batch = mm.sample_batch(music_streaming_data, batch_size=50)
    metrics = model.test_step(batch)

    assert metrics["loss_batch"] >= 0
    assert set(list(metrics.keys())) == set(
        [
            "loss",
            "precision",
            "recall",
            "binary_accuracy",
            "auc",
            "regularization_loss",
            "loss_batch",
            "weighted_auc",
        ]
    )


@testing_utils.mark_run_eagerly_modes
def test_column_based_sample_weight_check_loss_weighted_metrics(
    music_streaming_data: Dataset, run_eagerly: bool
):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.BinaryOutput(
        "like",
        post=mm.ColumnBasedSampleWeight(weight_column_name="click"),
    )

    model = mm.Model(inputs, mm.MLPBlock([8]), output_block)

    model.compile(
        optimizer="adam",
        run_eagerly=run_eagerly,
        weighted_metrics=["binary_accuracy"],
    )

    batch = mm.sample_batch(music_streaming_data, batch_size=50)
    metrics = model.test_step(batch)

    assert metrics["loss"] >= 0
    assert set(list(metrics.keys())) == set(
        [
            "loss",
            "precision",
            "recall",
            "binary_accuracy",
            "auc",
            "regularization_loss",
            "loss_batch",
            "weighted_binary_accuracy",
        ]
    )

    batch[1]["click"] = tf.ones_like(batch[1]["click"])
    model.compiled_metrics.reset_state()
    metrics_sample_weight_all_ones = model.test_step(batch)
    assert metrics_sample_weight_all_ones["loss_batch"] > metrics["loss_batch"]

    batch[1]["click"] = tf.zeros_like(batch[1]["click"])
    model.compiled_metrics.reset_state()
    metrics_sample_weight_all_zeros = model.test_step(batch)
    assert metrics_sample_weight_all_zeros["loss_batch"] == 0.0

    # Regular metrics are not affected by sample_weigth
    assert (
        metrics["binary_accuracy"]
        == metrics_sample_weight_all_zeros["binary_accuracy"]
        == metrics_sample_weight_all_ones["binary_accuracy"]
    )
    # But weighted metrics are different
    assert (
        metrics["weighted_binary_accuracy"]
        != metrics_sample_weight_all_ones["weighted_binary_accuracy"]
    )
    assert (
        metrics["weighted_binary_accuracy"]
        != metrics_sample_weight_all_zeros["weighted_binary_accuracy"]
    )
    assert (
        metrics_sample_weight_all_ones["weighted_binary_accuracy"]
        != metrics_sample_weight_all_zeros["weighted_binary_accuracy"]
    )


@testing_utils.mark_run_eagerly_modes
def test_column_based_sample_weight_with_multitask(
    music_streaming_data: Dataset, run_eagerly: bool
):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(
        music_streaming_data.schema,
        model_outputs={
            "click/binary_output": mm.BinaryOutput(
                "click",
                post=mm.ColumnBasedSampleWeight(
                    binary_class_weights=(1.0, 10.0),
                ),
            ),
            "like/binary_output": mm.BinaryOutput(
                "like",
                post=mm.SequentialBlock(
                    # Cascaded sample weights
                    mm.ColumnBasedSampleWeight(weight_column_name="click"),
                    mm.ColumnBasedSampleWeight(
                        weight_column_name="like", binary_class_weights=(1.0, 5.0)
                    ),
                ),
            ),
            "play_percentage/regression_output": mm.RegressionOutput(
                "play_percentage",
                post=mm.ColumnBasedSampleWeight(weight_column_name="click"),
            ),
        },
    )

    model = mm.Model(inputs, mm.MLPBlock([8]), output_block)

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
            "auc",
        ),
        "like/binary_output": ("precision", "recall", "binary_accuracy", "auc"),
        "play_percentage/regression_output": (
            tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
        ),
    }

    _, losses = testing_utils.model_test(
        model,
        music_streaming_data,
        run_eagerly=run_eagerly,
        reload_model=True,
        loss_weights=loss_weights,
        weighted_metrics=weighted_metrics,
    )

    assert losses.history["loss"][0] >= 0
    assert set(losses.history.keys()) == set(
        [
            "loss",
            "loss_batch",
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
        ]
    )


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "task_blocks",
    [None, mm.MLPBlock([32])],
)
@pytest.mark.parametrize(
    "enable_gate_weights_metrics",
    [False, True],
)
def test_mmoe_model(
    music_streaming_data: Dataset,
    run_eagerly: bool,
    task_blocks: Optional[Block],
    enable_gate_weights_metrics: bool,
):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(music_streaming_data.schema, task_blocks=task_blocks)
    num_experts = 4
    mmoe = mm.MMOEBlock(
        output_block,
        expert_block=mm.MLPBlock([64]),
        num_experts=num_experts,
        gate_block=mm.MLPBlock([32]),
        enable_gate_weights_metrics=enable_gate_weights_metrics,
    )
    model = mm.Model(inputs, mmoe, output_block)

    loss_weights = {
        "click/binary_output": 1.0,
        "play_percentage/regression_output": 2.0,
        "like/binary_output": 3.0,
    }

    _, losses = testing_utils.model_test(
        model,
        music_streaming_data,
        run_eagerly=run_eagerly,
        reload_model=True,
        loss_weights=loss_weights,
    )

    assert losses.history["loss"][0] >= 0

    expected_metrics = [
        "loss",
        "loss_batch",
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
        "regularization_loss",
    ]
    if enable_gate_weights_metrics:
        for task in loss_weights.keys():
            for i in range(num_experts):
                gate_metric_name = f"gate_{task}_weight_{i}"
                expected_metrics.append(gate_metric_name)

    assert set(losses.history.keys()) == set(expected_metrics)


@tf.keras.utils.register_keras_serializable(package="merlin_models")
class CustomSampleWeight(Block):
    def call(
        self,
        inputs,
        targets=None,
        features=None,
        target_name=None,
        training=False,
        testing=False,
        **kwargs,
    ) -> mm.Prediction:
        if not (training or testing) or targets is None:
            return inputs
        return mm.Prediction(inputs, targets[target_name], sample_weight=targets["click"])

    def compute_output_shape(self, input_shape):
        return input_shape


@testing_utils.mark_run_eagerly_modes
def test_mmoe_block_task_specific_sample_weight_and_weighted_metrics(
    music_streaming_data: Dataset, run_eagerly: bool
):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(
        music_streaming_data.schema,
        model_outputs={"like/binary_output": mm.BinaryOutput("like", post=CustomSampleWeight())},
    )

    mmoe = mm.MMOEBlock(output_block, expert_block=mm.MLPBlock([64]), num_experts=4)
    model = mm.Model(inputs, mmoe, output_block)

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
            "auc",
        ),
        "like/binary_output": ("precision", "recall", "binary_accuracy", "auc"),
        "play_percentage/regression_output": (
            tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
        ),
    }

    _, losses = testing_utils.model_test(
        model,
        music_streaming_data,
        run_eagerly=run_eagerly,
        reload_model=True,
        loss_weights=loss_weights,
        weighted_metrics=weighted_metrics,
    )

    assert losses.history["loss"][0] >= 0

    assert set(losses.history.keys()) == set(
        [
            "loss",
            "loss_batch",
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
        ]
    )


@testing_utils.mark_run_eagerly_modes
def test_mmoe_model_serialization(music_streaming_data: Dataset, run_eagerly: bool):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(
        music_streaming_data.schema,
    )
    num_experts = 4
    mmoe = mm.MMOEBlock(
        output_block,
        expert_block=mm.MLPBlock([64]),
        num_experts=num_experts,
        gate_block=mm.MLPBlock([26]),
    )

    weighted_metrics = {
        "click/binary_output": (
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            "auc",
        ),
        "like/binary_output": ("precision", "recall", "binary_accuracy", "auc"),
        "play_percentage/regression_output": (
            tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
        ),
    }

    model = mm.Model(inputs, mmoe, output_block)

    _, losses = testing_utils.model_test(
        model,
        music_streaming_data,
        run_eagerly=run_eagerly,
        reload_model=True,
        weighted_metrics=weighted_metrics,
    )

    assert losses.history["loss"][0] >= 0

    testing_utils.model_test(
        model,
        music_streaming_data,
        run_eagerly=run_eagerly,
        reload_model=True,
    )


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "task_blocks",
    [None, mm.MLPBlock([32])],
)
def test_cgc_model(music_streaming_data: Dataset, run_eagerly: bool, task_blocks: Optional[Block]):
    schema = music_streaming_data.schema
    inputs = mm.InputBlockV2(schema)
    output_block = mm.OutputBlock(music_streaming_data.schema, task_blocks=task_blocks)
    cgc = mm.CGCBlock(
        output_block,
        expert_block=mm.MLPBlock([64]),
        num_task_experts=2,
        num_shared_experts=3,
    )
    model = mm.Model(inputs, cgc, output_block)

    _, losses = testing_utils.model_test(
        model,
        music_streaming_data,
        run_eagerly=run_eagerly,
        reload_model=True,
    )

    assert losses.history["loss"][0] >= 0

    assert set(losses.history.keys()) == set(
        [
            "loss",
            "loss_batch",
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
            "regularization_loss",
        ]
    )


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "task_blocks",
    [None, mm.MLPBlock([32])],
)
def test_ple_model(music_streaming_data: Dataset, run_eagerly: bool, task_blocks: Optional[Block]):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(music_streaming_data.schema, task_blocks=task_blocks)
    ple = mm.PLEBlock(
        num_layers=2,
        outputs=output_block,
        expert_block=mm.MLPBlock([64]),
        num_task_experts=2,
        num_shared_experts=3,
    )
    model = mm.Model(inputs, ple, output_block)
    _, losses = testing_utils.model_test(
        model,
        music_streaming_data,
        run_eagerly=run_eagerly,
        reload_model=True,
    )

    assert losses.history["loss"][0] >= 0
    assert set(losses.history.keys()) == set(
        [
            "loss",
            "loss_batch",
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
            "regularization_loss",
        ]
    )


@testing_utils.mark_run_eagerly_modes
def test_ple_model_serialization(music_streaming_data: Dataset, run_eagerly: bool):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    output_block = mm.OutputBlock(music_streaming_data.schema, task_blocks=mm.MLPBlock([32]))
    ple = mm.PLEBlock(
        num_layers=2,
        outputs=output_block,
        expert_block=mm.MLPBlock([64]),
        num_task_experts=2,
        num_shared_experts=3,
        gate_block=mm.MLPBlock([16]),
    )
    model = mm.Model(inputs, ple, output_block)
    _, losses = testing_utils.model_test(
        model,
        music_streaming_data,
        run_eagerly=run_eagerly,
        reload_model=True,
    )

    assert losses.history["loss"][0] >= 0
