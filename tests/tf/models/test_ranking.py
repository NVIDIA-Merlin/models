# #
# # Copyright (c) 2021, NVIDIA CORPORATION.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
#
import pytest

import merlin.models.tf as ml
from merlin.models.data.synthetic import SyntheticData
from merlin.models.tf.utils import testing_utils


# TODO: Fix this test when `run_eagerly=False`
# @pytest.mark.parametrize("run_eagerly", [True, False])
def test_simple_model(ecommerce_data: SyntheticData, num_epochs=5, run_eagerly=True):
    body = ml.InputBlock(ecommerce_data.schema).connect(ml.MLPBlock([64]))
    model = body.connect(ml.BinaryClassificationTask("click"))
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(ecommerce_data.dataset, batch_size=50, epochs=num_epochs)
    metrics = model.evaluate(*ecommerce_data.tf_features_and_targets, return_dict=True)
    testing_utils.assert_binary_classification_loss_metrics(
        losses, metrics, target_name="click", num_epochs=num_epochs
    )


def test_dlrm_model_single_task_from_pred_task(ecommerce_data, num_epochs=5, run_eagerly=True):
    model = ml.DLRMModel(
        ecommerce_data.schema,
        embedding_dim=64,
        bottom_block=ml.MLPBlock([64]),
        top_block=ml.MLPBlock([32]),
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(ecommerce_data.dataset, batch_size=50, epochs=num_epochs)
    metrics = model.evaluate(*ecommerce_data.tf_features_and_targets, return_dict=True)
    testing_utils.assert_binary_classification_loss_metrics(
        losses, metrics, target_name="click", num_epochs=num_epochs
    )


@pytest.mark.parametrize("stacked", [True, False])
def test_dcn_model_single_task_from_pred_task(
    ecommerce_data, stacked, num_epochs=5, run_eagerly=True
):
    model = ml.DCNModel(
        ecommerce_data.schema,
        depth=3,
        deep_block=ml.MLPBlock([64]),
        stacked=stacked,
        prediction_tasks=ml.BinaryClassificationTask("click"),
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(ecommerce_data.dataset, batch_size=50, epochs=num_epochs)
    metrics = model.evaluate(*ecommerce_data.tf_features_and_targets, return_dict=True)
    testing_utils.assert_binary_classification_loss_metrics(
        losses, metrics, target_name="click", num_epochs=num_epochs
    )


def test_dlrm_model_single_head_multiple_tasks(
    music_streaming_data, num_epochs=5, run_eagerly=True
):
    dlrm_body = ml.DLRMBlock(
        music_streaming_data.schema,
        embedding_dim=64,
        bottom_block=ml.MLPBlock([64]),
        top_block=ml.MLPBlock([32]),
    )

    tasks_blocks = dict(click=ml.MLPBlock([16]), play_percentage=ml.MLPBlock([20]))

    prediction_tasks = ml.PredictionTasks(
        music_streaming_data.schema,
        task_blocks=tasks_blocks,
        task_weight_dict={"click": 2.0, "play_percentage": 1.0},
    )

    model = dlrm_body.connect(ml.MLPBlock([64]), prediction_tasks)

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data.dataset, batch_size=50, epochs=num_epochs)
    metrics = model.evaluate(*music_streaming_data.tf_features_and_targets, return_dict=True)
    testing_utils.assert_binary_classification_loss_metrics(
        losses, metrics, target_name="click", num_epochs=num_epochs
    )
    testing_utils.assert_regression_loss_metrics(
        losses, metrics, target_name="play_percentage", num_epochs=num_epochs
    )


@pytest.mark.parametrize("prediction_task", [ml.BinaryClassificationTask, ml.RegressionTask])
def test_serialization_model(ecommerce_data: SyntheticData, prediction_task):
    body = ml.InputBlock(ecommerce_data.schema).connect(ml.MLPBlock([64]))
    model = body.connect(prediction_task("click"))

    copy_model = testing_utils.assert_serialization(model)
    testing_utils.assert_loss_and_metrics_are_valid(
        copy_model, ecommerce_data.tf_features_and_targets
    )


@pytest.mark.parametrize("prediction_task", [None, ml.BinaryClassificationTask, ml.RegressionTask])
@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("model_name", ["mlp", "dlrm"])
def test_resume_training(ecommerce_data: SyntheticData, prediction_task, run_eagerly, model_name):
    from merlin.models.tf.utils import testing_utils

    if prediction_task:
        prediction_task = prediction_task("click")
    else:
        # Do multi-task learning if no prediction task is provided
        prediction_task = ml.PredictionTasks(ecommerce_data.schema)

    if model_name == "dlrm":
        body = ml.DLRMBlock(ecommerce_data.schema, embedding_dim=64, bottom_block=ml.MLPBlock([64]))
    else:
        body = ml.InputBlock(ecommerce_data.schema).connect(ml.MLPBlock([64]))
    model = body.connect(prediction_task)

    dataset = ecommerce_data.tf_dataloader(batch_size=50)
    copy_model = testing_utils.assert_model_is_retrainable(model, dataset, run_eagerly=run_eagerly)

    assert copy_model is not None
