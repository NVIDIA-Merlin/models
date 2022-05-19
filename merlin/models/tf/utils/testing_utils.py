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
from typing import Any, Tuple

import numpy as np
import pytest
import tensorflow as tf

import merlin.io
from merlin.models.tf.models.base import Model


def mark_run_eagerly_modes(*args, **kwargs):
    modes = [True, False]

    # As of TF 2.5 there's a bug that our EmbeddingFeatures don't work on M1 Macs
    if "macOS" in platform.platform() and "arm64-arm-64bit" in platform.platform():
        modes = [True]

    return pytest.mark.parametrize("run_eagerly", modes)(*args, **kwargs)


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


def model_test(
    model: Model,
    dataset: merlin.io.Dataset,
    run_eagerly: bool = True,
    optimizer="adam",
    epochs: int = 1,
    **kwargs,
) -> Tuple[Model, Any]:
    """Generic model test. It will compile & fit the model and make sure it can be re-trained."""

    model.compile(run_eagerly=run_eagerly, optimizer=optimizer, **kwargs)
    losses = model.fit(dataset, batch_size=50, epochs=epochs)

    assert len(losses.epoch) == epochs

    assert isinstance(model.from_config(model.get_config()), type(model))

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded_model = tf.keras.models.load_model(tmpdir)

    assert isinstance(loaded_model, type(model))

    np.array_equal(
        model.predict(dataset, batch_size=50), loaded_model.predict(dataset, batch_size=50)
    )

    loaded_model.compile(run_eagerly=run_eagerly, optimizer=optimizer, **kwargs)
    losses = loaded_model.fit(dataset, batch_size=50, epochs=epochs)

    assert len(losses.epoch) == epochs

    return loaded_model, losses
