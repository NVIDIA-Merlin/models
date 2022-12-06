from typing import Dict, Optional

import pytest
import tensorflow as tf

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
        dict(binary_classification_task=mm.MLPBlock([16]), regression_task=mm.MLPBlock([20])),
        {
            "click/binary_classification_task": mm.MLPBlock([16]),
            "play_percentage/regression_task": mm.MLPBlock([20]),
        },
    ],
)
def test_model_with_multiple_tasks(music_streaming_data: Dataset, task_blocks, run_eagerly: bool):
    music_streaming_data.schema = music_streaming_data.schema.without("like")

    inputs = mm.InputBlock(music_streaming_data.schema)
    prediction_tasks = mm.PredictionTasks(music_streaming_data.schema, task_blocks=task_blocks)
    model = mm.Model(inputs, mm.MLPBlock([64]), prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(mm.sample_batch(music_streaming_data, batch_size=50))

    assert metrics["loss"] >= 0
    assert set(list(metrics.keys())) == set(
        [
            "loss",
            "loss_batch",
            "regularization_loss",
            "click/binary_classification_task_loss",
            "play_percentage/regression_task_loss",
            "play_percentage/regression_task_root_mean_squared_error",
            "click/binary_classification_task_precision",
            "click/binary_classification_task_recall",
            "click/binary_classification_task_binary_accuracy",
            "click/binary_classification_task_auc",
        ]
    )
    if task_blocks:
        assert model.prediction_tasks[0].task_block != model.prediction_tasks[1].task_block


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

    metrics = model.train_step(mm.sample_batch(music_streaming_data, batch_size=50))

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
        "like/binary_classification_task_precision_1",
        "like/binary_classification_task_recall_1",
        "like/binary_classification_task_binary_accuracy",
        "like/binary_classification_task_auc_1",
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
            outputs = outputs.copy_with_updates(sample_weight=targets)
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

    metrics = model.train_step(mm.sample_batch(music_streaming_data, batch_size=50))

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
            "like/binary_classification_task_auc_1",
            "like/binary_classification_task_binary_accuracy",
            "like/binary_classification_task_loss",
            "like/binary_classification_task_precision_1",
            "like/binary_classification_task_recall_1",
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
        schema=schema,
    )
    model = mm.Model(inputs, cgc, prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(mm.sample_batch(music_streaming_data, batch_size=50))

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
            "like/binary_classification_task_precision_1",
            "like/binary_classification_task_recall_1",
            "like/binary_classification_task_binary_accuracy",
            "like/binary_classification_task_auc_1",
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

    metrics = model.train_step(mm.sample_batch(music_streaming_data, batch_size=50))

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
            "like/binary_classification_task_precision_1",
            "like/binary_classification_task_recall_1",
            "like/binary_classification_task_binary_accuracy",
            "like/binary_classification_task_auc_1",
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
