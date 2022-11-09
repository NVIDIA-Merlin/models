from typing import Dict

import pytest
import tensorflow as tf

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.models.tf.core.base import Block, PredictionOutput
from merlin.models.tf.utils import testing_utils


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize(
    "task_blocks",
    [
        None,
        ml.MLPBlock([32]),
        dict(click=ml.MLPBlock([16]), play_percentage=ml.MLPBlock([20])),
        dict(binary_classification_task=ml.MLPBlock([16]), regression_task=ml.MLPBlock([20])),
        {
            "click/binary_classification_task": ml.MLPBlock([16]),
            "play_percentage/regression_task": ml.MLPBlock([20]),
        },
    ],
)
def test_model_with_multiple_tasks(music_streaming_data: Dataset, task_blocks, run_eagerly: bool):
    music_streaming_data.schema = music_streaming_data.schema.without("like")

    inputs = ml.InputBlock(music_streaming_data.schema)
    prediction_tasks = ml.PredictionTasks(music_streaming_data.schema, task_blocks=task_blocks)
    model = ml.Model(inputs, ml.MLPBlock([64]), prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(ml.sample_batch(music_streaming_data, batch_size=50))

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
def test_mmoe_head(music_streaming_data: Dataset, run_eagerly: bool):
    inputs = ml.InputBlock(music_streaming_data.schema)
    prediction_tasks = ml.PredictionTasks(music_streaming_data.schema)
    mmoe = ml.MMOEBlock(prediction_tasks, expert_block=ml.MLPBlock([64]), num_experts=4)
    model = ml.Model(inputs, ml.MLPBlock([64]), mmoe, prediction_tasks)

    loss_weights = {
        "click/binary_classification_task": 1.0,
        "like/binary_classification_task": 2.0,
        "play_percentage/regression_task": 3.0,
    }

    model.compile(optimizer="adam", run_eagerly=run_eagerly, loss_weights=loss_weights)

    metrics = model.train_step(ml.sample_batch(music_streaming_data, batch_size=50))

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
def test_mmoe_head_task_specific_sample_weight_and_weighted_metrics(
    music_streaming_data: Dataset, run_eagerly: bool
):
    class CustomSampleWeight(Block):
        def call_outputs(
            self,
            outputs: PredictionOutput,
            features: Dict[str, tf.Tensor] = None,
            targets: Dict[str, tf.Tensor] = None,
            training=True,
            testing=False,
            **kwargs,
        ) -> PredictionOutput:
            # Computes loss for the like loss only for clicked items
            outputs = outputs.copy_with_updates(sample_weight=targets["click"])
            return outputs

    inputs = ml.InputBlock(music_streaming_data.schema)
    prediction_tasks = ml.PredictionTasks(
        music_streaming_data.schema, task_pre_dict={"like": CustomSampleWeight()}
    )
    mmoe = ml.MMOEBlock(prediction_tasks, expert_block=ml.MLPBlock([64]), num_experts=4)
    model = ml.Model(inputs, ml.MLPBlock([64]), mmoe, prediction_tasks)

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

    metrics = model.train_step(ml.sample_batch(music_streaming_data, batch_size=50))

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
def test_ple_head(music_streaming_data: Dataset, run_eagerly: bool):
    inputs = ml.InputBlock(music_streaming_data.schema)
    prediction_tasks = ml.PredictionTasks(music_streaming_data.schema)
    cgc = ml.CGCBlock(
        prediction_tasks, expert_block=ml.MLPBlock([64]), num_task_experts=2, num_shared_experts=2
    )
    model = ml.Model(inputs, ml.MLPBlock([64]), cgc, prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    metrics = model.train_step(ml.sample_batch(music_streaming_data, batch_size=50))

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
