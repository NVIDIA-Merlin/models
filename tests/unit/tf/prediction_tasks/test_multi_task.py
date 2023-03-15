from typing import Dict, Optional

import pytest
import tensorflow as tf
from tensorflow.keras.metrics import Metric

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.core.base import Block, PredictionOutput
from merlin.models.tf.utils import testing_utils


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "task_blocks",
    [
        None,
        mm.MLPBlock([32]),
        dict(click=mm.MLPBlock([16]), play_percentage=mm.MLPBlock([20])),
        {
            "click/binary_classification_task": mm.MLPBlock([16]),
            "play_percentage/regression_task": mm.MLPBlock([20]),
        },
    ],
)
def test_model_with_multiple_tasks_with_task_towers(
    music_streaming_data: Dataset, task_blocks, run_eagerly: bool
):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    prediction_tasks = mm.PredictionTasks(music_streaming_data.schema, task_blocks=task_blocks)
    model = mm.Model(inputs, mm.MLPBlock([64]), prediction_tasks)
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
            "click/binary_classification_task_loss",
            "click/binary_classification_task_precision",
            "click/binary_classification_task_recall",
            "click/binary_classification_task_binary_accuracy",
            "click/binary_classification_task_auc",
            "like/binary_classification_task_loss",
            "like/binary_classification_task_precision",
            "like/binary_classification_task_recall",
            "like/binary_classification_task_binary_accuracy",
            "like/binary_classification_task_auc",
            "play_percentage/regression_task_loss",
            "play_percentage/regression_task_root_mean_squared_error",
        ]
    )
    if task_blocks:
        assert model.prediction_tasks[0].task_block != model.prediction_tasks[2].task_block
        if isinstance(task_blocks, dict):
            # Ensures for like there is no task tower
            assert model.prediction_tasks[1].task_block is None
        else:
            assert model.prediction_tasks[0].task_block != model.prediction_tasks[1].task_block


@pytest.mark.parametrize("run_eagerly", [False])
@pytest.mark.parametrize(
    "metrics",
    [
        {
            "click/binary_classification_task": (
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ),
            "like/binary_classification_task": [
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")
            ],
        },
        None,
        tf.keras.metrics.AUC(name="auc"),
        (tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")),
    ],
)
def test_model_with_multiple_tasks_metrics(
    music_streaming_data: Dataset, run_eagerly: bool, metrics
):
    music_streaming_data.schema = music_streaming_data.schema.without("play_percentage")

    inputs = mm.InputBlockV2(music_streaming_data.schema)
    prediction_tasks = mm.PredictionTasks(music_streaming_data.schema)
    model = mm.Model(inputs, mm.MLPBlock([64]), prediction_tasks)

    weighted_metrics = metrics

    expected_metrics = [
        "loss",
        "regularization_loss",
        "loss_batch",
        "click/binary_classification_task_loss",
        "like/binary_classification_task_loss",
    ]
    if isinstance(metrics, Metric):
        expected_metrics.extend(
            [
                "click/binary_classification_task_auc",
                "like/binary_classification_task_auc",
                "click/binary_classification_task_weighted_auc",
                "like/binary_classification_task_weighted_auc",
            ]
        )

        # Creating another copy of the metrics to avoid error for reusing same metrics
        # for metrics and weighted_metrics
        weighted_metrics = (tf.keras.metrics.AUC(name="auc"),)
    elif isinstance(metrics, (list, tuple)):
        expected_metrics.extend(
            [
                "click/binary_classification_task_precision",
                "like/binary_classification_task_precision",
                "click/binary_classification_task_recall",
                "like/binary_classification_task_recall",
                "click/binary_classification_task_weighted_precision",
                "like/binary_classification_task_weighted_precision",
                "click/binary_classification_task_weighted_recall",
                "like/binary_classification_task_weighted_recall",
            ]
        )
    elif isinstance(metrics, dict):
        expected_metrics.extend(
            [
                "click/binary_classification_task_precision",
                "click/binary_classification_task_recall",
                "like/binary_classification_task_binary_accuracy",
                "click/binary_classification_task_weighted_precision",
                "click/binary_classification_task_weighted_recall",
                "like/binary_classification_task_weighted_binary_accuracy",
            ]
        )
        # Creating another copy of the metrics to avoid error for reusing same metrics
        # for metrics and weighted_metrics
        weighted_metrics = {
            "click/binary_classification_task": (
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ),
            "like/binary_classification_task": [
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")
            ],
        }
    elif metrics is None:
        # Use default metrics
        expected_metrics.extend(
            [
                "click/binary_classification_task_precision",
                "click/binary_classification_task_recall",
                "click/binary_classification_task_binary_accuracy",
                "click/binary_classification_task_auc",
                "like/binary_classification_task_precision",
                "like/binary_classification_task_recall",
                "like/binary_classification_task_binary_accuracy",
                "like/binary_classification_task_auc",
            ]
        )

    model.compile(
        optimizer="adam",
        run_eagerly=run_eagerly,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
        from_serialized=False,
    )

    metrics_results = model.train_step(
        mm.sample_batch(music_streaming_data, batch_size=50, prepare_features=False)
    )

    assert metrics_results["loss"] >= 0
    assert set(metrics_results.keys()) == set(expected_metrics)


@testing_utils.mark_run_eagerly_modes
def test_model_with_multiple_tasks_loss_weights_and_weighted_metrics(
    music_streaming_data: Dataset, run_eagerly: bool
):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    prediction_tasks = mm.PredictionTasks(music_streaming_data.schema)
    model = mm.Model(inputs, mm.MLPBlock([64]), prediction_tasks)

    loss_weights = {
        "click/binary_classification_task": 1.0,
        "play_percentage/regression_task": 2.0,
        "like/binary_classification_task": 3.0,
    }

    weighted_metrics = {
        "click/binary_classification_task": (
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ),
        "like/binary_classification_task": (
            tf.keras.metrics.Precision(name="weighted_precision"),
            tf.keras.metrics.Recall(name="weighted_recall"),
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="weighted_auc"),
        ),
        "play_percentage/regression_task": (
            tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
        ),
    }

    model.compile(
        optimizer="adam",
        run_eagerly=run_eagerly,
        loss_weights=loss_weights,
        weighted_metrics=weighted_metrics,
    )

    batch = mm.sample_batch(music_streaming_data, batch_size=50, prepare_features=False)

    metrics = model.test_step(batch)

    assert metrics["loss"] >= 0
    assert set(metrics.keys()) == set(
        [
            "loss",
            "click/binary_classification_task_loss",
            "like/binary_classification_task_loss",
            "play_percentage/regression_task_loss",
            "click/binary_classification_task_precision",
            "click/binary_classification_task_recall",
            "click/binary_classification_task_binary_accuracy",
            "click/binary_classification_task_auc",
            "click/binary_classification_task_weighted_precision",
            "click/binary_classification_task_weighted_recall",
            "click/binary_classification_task_weighted_binary_accuracy",
            "click/binary_classification_task_weighted_auc",
            "like/binary_classification_task_precision",
            "like/binary_classification_task_recall",
            "like/binary_classification_task_binary_accuracy",
            "like/binary_classification_task_auc",
            "like/binary_classification_task_weighted_precision",
            "like/binary_classification_task_weighted_recall",
            "like/binary_classification_task_weighted_binary_accuracy",
            "like/binary_classification_task_weighted_auc",
            "play_percentage/regression_task_root_mean_squared_error",
            "play_percentage/regression_task_weighted_root_mean_squared_error",
            "regularization_loss",
            "loss_batch",
        ]
    )

    metrics2 = model.test_step(batch)
    assert metrics["loss"] == metrics2["loss"]
    for m in metrics:
        assert metrics[m] == metrics2[m]

    # Disabling losses weights
    model.compile(loss_weights=None)
    metrics_non_weighted = model.test_step(batch)

    assert metrics_non_weighted["loss"] != metrics["loss"]
    assert (
        metrics_non_weighted["click/binary_classification_task_loss"]
        == metrics["click/binary_classification_task_loss"]
    )
    assert (
        metrics_non_weighted["like/binary_classification_task_loss"]
        == metrics["like/binary_classification_task_loss"]
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
    prediction_tasks = mm.PredictionTasks(music_streaming_data.schema, task_blocks=task_blocks)
    num_experts = 4
    mmoe = mm.MMOEBlock(
        prediction_tasks,
        expert_block=mm.MLPBlock([64]),
        num_experts=num_experts,
        gate_block=mm.MLPBlock([32]),
        enable_gate_weights_metrics=enable_gate_weights_metrics,
    )
    model = mm.Model(inputs, mmoe, prediction_tasks)

    loss_weights = {
        "click/binary_classification_task": 1.0,
        "like/binary_classification_task": 2.0,
        "play_percentage/regression_task": 3.0,
    }

    model.compile(optimizer="adam", run_eagerly=run_eagerly, loss_weights=loss_weights)

    metrics = model.train_step(
        mm.sample_batch(music_streaming_data, batch_size=50, prepare_features=False)
    )

    assert metrics["loss"] >= 0

    expected_metrics = [
        "loss",
        "loss_batch",
        "click/binary_classification_task_loss",
        "like/binary_classification_task_loss",
        "play_percentage/regression_task_loss",
        "click/binary_classification_task_precision",
        "click/binary_classification_task_recall",
        "click/binary_classification_task_binary_accuracy",
        "click/binary_classification_task_auc",
        "like/binary_classification_task_precision",
        "like/binary_classification_task_recall",
        "like/binary_classification_task_binary_accuracy",
        "like/binary_classification_task_auc",
        "play_percentage/regression_task_root_mean_squared_error",
        "regularization_loss",
    ]
    if enable_gate_weights_metrics:
        for task in loss_weights.keys():
            for i in range(num_experts):
                gate_metric_name = f"gate_{task}_weight_{i}"
                expected_metrics.append(gate_metric_name)

    assert set(metrics.keys()) == set(expected_metrics)


@testing_utils.mark_run_eagerly_modes
def test_mmoe_block_task_specific_sample_weight_and_weighted_metrics(
    music_streaming_data: Dataset, run_eagerly: bool
):
    class CustomSampleWeight(Block):
        def call_outputs(
            self,
            outputs: PredictionOutput,
            features: Dict[str, tf.Tensor] = None,
            targets: tf.Tensor = None,
            training=True,
            testing=False,
            **kwargs,
        ) -> PredictionOutput:
            # Computes loss for the like loss only for clicked items
            outputs = outputs.copy_with_updates(
                sample_weight=tf.expand_dims(targets["click"], -1),
            )
            return outputs

    inputs = mm.InputBlockV2(music_streaming_data.schema)
    prediction_tasks = mm.PredictionTasks(
        music_streaming_data.schema, task_pre_dict={"like": CustomSampleWeight()}
    )
    mmoe = mm.MMOEBlock(prediction_tasks, expert_block=mm.MLPBlock([64]), num_experts=4)
    model = mm.Model(inputs, mmoe, prediction_tasks)

    loss_weights = {
        "click/binary_classification_task": 1.0,
        "like/binary_classification_task": 2.0,
        "play_percentage/regression_task": 3.0,
    }

    weighted_metrics = {
        "click/binary_classification_task": (
            tf.keras.metrics.Precision(name="weighted_precision"),
            tf.keras.metrics.Recall(name="weighted_recall"),
            tf.keras.metrics.BinaryAccuracy(name="weighted_binary_accuracy"),
            tf.keras.metrics.AUC(name="weighted_auc"),
        ),
        "like/binary_classification_task": (
            tf.keras.metrics.Precision(name="weighted_precision"),
            tf.keras.metrics.Recall(name="weighted_recall"),
            tf.keras.metrics.BinaryAccuracy(name="weighted_binary_accuracy"),
            tf.keras.metrics.AUC(name="weighted_auc"),
        ),
        "play_percentage/regression_task": (
            tf.keras.metrics.RootMeanSquaredError(name="weighted_root_mean_squared_error"),
        ),
    }

    model.compile(
        optimizer="adam",
        run_eagerly=run_eagerly,
        loss_weights=loss_weights,
        weighted_metrics=weighted_metrics,
    )

    metrics = model.train_step(
        mm.sample_batch(music_streaming_data, batch_size=50, prepare_features=False)
    )

    assert metrics["loss"] >= 0
    assert set(metrics.keys()) == set(
        [
            "loss",
            "loss_batch",
            "regularization_loss",
            "click/binary_classification_task_auc",
            "click/binary_classification_task_binary_accuracy",
            "click/binary_classification_task_loss",
            "click/binary_classification_task_precision",
            "click/binary_classification_task_recall",
            "click/binary_classification_task_weighted_auc",
            "click/binary_classification_task_weighted_binary_accuracy",
            "click/binary_classification_task_weighted_precision",
            "click/binary_classification_task_weighted_recall",
            "like/binary_classification_task_auc",
            "like/binary_classification_task_binary_accuracy",
            "like/binary_classification_task_loss",
            "like/binary_classification_task_precision",
            "like/binary_classification_task_recall",
            "like/binary_classification_task_weighted_auc",
            "like/binary_classification_task_weighted_binary_accuracy",
            "like/binary_classification_task_weighted_precision",
            "like/binary_classification_task_weighted_recall",
            "play_percentage/regression_task_loss",
            "play_percentage/regression_task_root_mean_squared_error",
            "play_percentage/regression_task_weighted_root_mean_squared_error",
        ]
    )


@testing_utils.mark_run_eagerly_modes
def test_mmoe_model_serialization(music_streaming_data: Dataset, run_eagerly: bool):
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    prediction_tasks = mm.PredictionTasks(
        music_streaming_data.schema, task_blocks=mm.MLPBlock([32])
    )
    num_experts = 4
    mmoe = mm.MMOEBlock(
        prediction_tasks,
        expert_block=mm.MLPBlock([64]),
        num_experts=num_experts,
        gate_block=mm.MLPBlock([26]),
    )
    model = mm.Model(inputs, mmoe, prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    loader = mm.Loader(music_streaming_data, batch_size=8, shuffle=False)
    testing_utils.model_test(
        model,
        loader,
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
    prediction_tasks = mm.PredictionTasks(schema, task_blocks=task_blocks)
    cgc = mm.CGCBlock(
        prediction_tasks,
        expert_block=mm.MLPBlock([64]),
        num_task_experts=2,
        num_shared_experts=3,
    )
    model = mm.Model(inputs, cgc, prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(
        mm.sample_batch(music_streaming_data, batch_size=50, prepare_features=False)
    )

    assert metrics["loss"] >= 0
    assert set(metrics.keys()) == set(
        [
            "loss",
            "loss_batch",
            "click/binary_classification_task_loss",
            "like/binary_classification_task_loss",
            "play_percentage/regression_task_loss",
            "click/binary_classification_task_precision",
            "click/binary_classification_task_recall",
            "click/binary_classification_task_binary_accuracy",
            "click/binary_classification_task_auc",
            "like/binary_classification_task_precision",
            "like/binary_classification_task_recall",
            "like/binary_classification_task_binary_accuracy",
            "like/binary_classification_task_auc",
            "play_percentage/regression_task_root_mean_squared_error",
            "regularization_loss",
        ]
    )


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "task_blocks",
    [None, mm.MLPBlock([32])],
)
def test_ple_model(music_streaming_data: Dataset, run_eagerly: bool, task_blocks: Optional[Block]):
    schema = music_streaming_data.schema
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    prediction_tasks = mm.PredictionTasks(schema, task_blocks=task_blocks)
    cgc = mm.PLEBlock(
        num_layers=2,
        outputs=prediction_tasks,
        expert_block=mm.MLPBlock([64]),
        num_task_experts=2,
        num_shared_experts=3,
    )
    model = mm.Model(inputs, cgc, prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(
        mm.sample_batch(music_streaming_data, batch_size=50, prepare_features=False)
    )

    assert metrics["loss"] >= 0
    assert set(metrics.keys()) == set(
        [
            "loss",
            "click/binary_classification_task_loss",
            "like/binary_classification_task_loss",
            "play_percentage/regression_task_loss",
            "click/binary_classification_task_precision",
            "click/binary_classification_task_recall",
            "click/binary_classification_task_binary_accuracy",
            "click/binary_classification_task_auc",
            "like/binary_classification_task_precision",
            "like/binary_classification_task_recall",
            "like/binary_classification_task_binary_accuracy",
            "like/binary_classification_task_auc",
            "play_percentage/regression_task_root_mean_squared_error",
            "regularization_loss",
            "loss_batch",
        ]
    )


@testing_utils.mark_run_eagerly_modes
def test_ple_model_serialization(music_streaming_data: Dataset, run_eagerly: bool):
    schema = music_streaming_data.schema
    inputs = mm.InputBlockV2(music_streaming_data.schema)
    prediction_tasks = mm.PredictionTasks(schema, task_blocks=mm.MLPBlock([32]))
    cgc = mm.PLEBlock(
        num_layers=2,
        outputs=prediction_tasks,
        expert_block=mm.MLPBlock([64]),
        num_task_experts=2,
        num_shared_experts=3,
        gate_block=mm.MLPBlock([16]),
    )
    model = mm.Model(inputs, cgc, prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    loader = mm.Loader(music_streaming_data, batch_size=8, shuffle=False)
    testing_utils.model_test(
        model,
        loader,
        run_eagerly=run_eagerly,
        reload_model=True,
    )
