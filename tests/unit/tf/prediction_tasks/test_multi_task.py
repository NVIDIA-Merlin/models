import pytest

import merlin.models.tf as ml
from merlin.io import Dataset


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
def test_model_with_multiple_tasks(music_streaming_data: Dataset, task_blocks):
    music_streaming_data.schema = music_streaming_data.schema.without("like")

    inputs = ml.InputBlock(music_streaming_data.schema)
    prediction_tasks = ml.PredictionTasks(music_streaming_data.schema, task_blocks=task_blocks)
    model = ml.Model(inputs, ml.MLPBlock([64]), prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=True)

    metrics = model.train_step(ml.sample_batch(music_streaming_data, batch_size=50))

    assert metrics["loss"] >= 0
    assert set(list(metrics.keys())) == set(
        [
            "loss",
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


def test_mmoe_head(music_streaming_data: Dataset):
    inputs = ml.InputBlock(music_streaming_data.schema)
    prediction_tasks = ml.PredictionTasks(music_streaming_data.schema)
    mmoe = ml.MMOEBlock(prediction_tasks, expert_block=ml.MLPBlock([64]), num_experts=4)
    model = ml.Model(inputs, ml.MLPBlock([64]), mmoe, prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=True)

    metrics = model.train_step(ml.sample_batch(music_streaming_data, batch_size=50))

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
        ]
    )


def test_ple_head(music_streaming_data: Dataset):
    inputs = ml.InputBlock(music_streaming_data.schema)
    prediction_tasks = ml.PredictionTasks(music_streaming_data.schema)
    cgc = ml.CGCBlock(
        prediction_tasks, expert_block=ml.MLPBlock([64]), num_task_experts=2, num_shared_experts=2
    )
    model = ml.Model(inputs, ml.MLPBlock([64]), cgc, prediction_tasks)
    model.compile(optimizer="adam", run_eagerly=True)

    metrics = model.train_step(ml.sample_batch(music_streaming_data, batch_size=50))

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
        ]
    )
