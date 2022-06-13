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
from tensorflow.keras import regularizers

import merlin.models.tf as ml
from merlin.io import Dataset


@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("activation", ["relu", "tanh"])
@pytest.mark.parametrize("dropout", [None, 0.5])
@pytest.mark.parametrize(
    "normalization", [None, "batch_norm", tf.keras.layers.BatchNormalization()]
)
def test_mlp_block_yoochoose(
    testing_data: Dataset,
    dim,
    activation,
    dropout,
    normalization,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer="l2",
    bias_regularizer="l1",
    activity_regularizer=regularizers.l2(1e-4),
):
    inputs = ml.InputBlock(testing_data.schema)

    mlp = ml.MLPBlock(
        [dim],
        activation=activation,
        dropout=dropout,
        normalization=normalization,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
    )
    body = ml.SequentialBlock([inputs, mlp])

    outputs = body(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, dim]
    assert mlp.layers[0].units == dim
    assert mlp.layers[0].dense.activation.__name__ == activation

    if dropout:
        assert mlp.layers[1].rate == dropout
    if normalization:
        if normalization == "batch_norm":
            normalization = tf.keras.layers.BatchNormalization()

        assert mlp.layers[-1].__class__.__name__ == normalization.__class__.__name__


@pytest.mark.parametrize("no_activation_last_layer", [False, True])
@pytest.mark.parametrize("dims", [[32], [64, 32]])
def test_mlp_block_no_activation_last_layer(no_activation_last_layer, dims):
    mlp = ml.MLPBlock(dims, activation="relu", no_activation_last_layer=no_activation_last_layer)

    for idx, layer in enumerate(mlp.layers):
        if no_activation_last_layer and idx == len(mlp.layers) - 1:
            assert layer.dense.activation.__name__ == "linear"
        else:
            assert layer.dense.activation.__name__ == "relu"


@pytest.mark.parametrize("run_eagerly", [True])
def test_mlp_model_save(ecommerce_data: Dataset, run_eagerly: bool, tmp_path):
    model = ml.Model.from_block(
        ml.MLPBlock(
            [64], kernel_regularizer=regularizers.l2(1e-1), bias_regularizer=regularizers.l2(1e-1)
        ),
        ecommerce_data.schema,
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    model.fit(ecommerce_data, batch_size=50, epochs=1)
    model.save(str(tmp_path))

    copy_model = tf.keras.models.load_model(str(tmp_path))
    copy_model.compile(optimizer="adam", run_eagerly=run_eagerly)

    assert isinstance(copy_model, tf.keras.Model)
