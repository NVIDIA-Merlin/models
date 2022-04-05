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

import platform
import tempfile

import pytest
import tensorflow as tf

from merlin.models.tf.models.base import Model
from merlin.models.tf.prediction_tasks.classification import BinaryClassificationTask


def mark_run_eagerly_modes(*args, **kwargs):
    modes = [True, False]

    # As of TF 2.5 there's a bug that our EmbeddingFeatures don't work on M1 Macs
    if "macOS" in platform.platform() and "arm64-arm-64bit" in platform.platform():
        modes = [True]

    return pytest.mark.parametrize("run_eagerly", modes)(*args, **kwargs)


def assert_body_works_in_model(dataset, body, run_eagerly, num_epochs=5):
    model = body.connect(BinaryClassificationTask("click"))
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(dataset, batch_size=50, epochs=num_epochs)
    metrics = model.evaluate(dataset, batch_size=50, return_dict=True)

    assert_binary_classification_loss_metrics(
        losses, metrics, target_name="click", num_epochs=num_epochs
    )


def assert_binary_classification_loss_metrics(losses, metrics, target_name, num_epochs):
    metrics_names = [
        f"{target_name}/binary_classification_task/precision",
        f"{target_name}/binary_classification_task/recall",
        f"{target_name}/binary_classification_task/binary_accuracy",
        f"{target_name}/binary_classification_task/auc",
        "loss",
        "regularization_loss",
        "total_loss",
    ]

    assert len(set(metrics.keys()).intersection(set(metrics_names))) == len(metrics_names)

    assert len(set(losses.history.keys()).intersection(set(metrics_names))) == len(metrics_names)
    assert len(losses.epoch) == num_epochs
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list
        assert len(losses.history[metric]) == num_epochs

    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


def assert_regression_loss_metrics(losses, metrics, target_name, num_epochs):
    metrics_names = [
        f"{target_name}/regression_task/root_mean_squared_error",
        "loss",
        "regularization_loss",
        "total_loss",
    ]

    assert len(set(metrics.keys()).intersection(set(metrics_names))) == len(metrics_names)

    assert len(set(losses.history.keys()).intersection(set(metrics_names))) == len(metrics_names)
    assert len(losses.epoch) == num_epochs
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list
        assert len(losses.history[metric]) == num_epochs

    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


def assert_loss_and_metrics_are_valid(
    loss_block, features_and_targets, call_body=True, training=False
):
    features, targets = features_and_targets
    predictions = loss_block(features, training=training)
    loss = loss_block.compute_loss(predictions, targets, call_body=call_body, training=training)
    # metrics = input.metric_results()

    assert loss is not None
    # assert len(metrics) == len(input.metrics)


def assert_serialization(layer):
    serialized = layer.get_config()
    copy_layer = layer.from_config(serialized)

    assert isinstance(copy_layer, layer.__class__)

    return copy_layer


def assert_model_is_retrainable(
    model: Model, data, run_eagerly: bool = True, optimizer="adam", **kwargs
):
    model.compile(run_eagerly=run_eagerly, optimizer=optimizer, **kwargs)
    losses = model.fit(data, batch_size=50, epochs=1)

    assert len(losses.epoch) == 1
    # assert all(0 <= loss <= 1 for loss in losses.history["loss"])

    assert model.from_config(model.get_config()) is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded_model = tf.keras.models.load_model(tmpdir)

    assert isinstance(loaded_model, Model)
    loaded_model.compile(run_eagerly=run_eagerly, optimizer=optimizer, **kwargs)
    losses = loaded_model.fit(data, batch_size=50, epochs=1)

    assert len(losses.epoch) == 1
    # assert all(0 <= loss <= 1 for loss in losses.history["loss"])

    return loaded_model
