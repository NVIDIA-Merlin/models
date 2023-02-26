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
from merlin.models.tf.core.aggregation import SequenceAggregator
from merlin.models.tf.loader import Loader
from merlin.models.tf.utils import testing_utils
from merlin.schema.tags import Tags


@pytest.mark.parametrize("dim", [32, 64])
@pytest.mark.parametrize("activation", ["relu", "tanh"])
@pytest.mark.parametrize("dropout", [None, 0.5])
@pytest.mark.parametrize(
    "normalization", [None, "batch_norm", tf.keras.layers.BatchNormalization()]
)
def test_mlp_block(
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


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_mlp_model_with_sequential_features_and_combiner(
    sequence_testing_data: Dataset, run_eagerly
):
    schema = sequence_testing_data.schema
    target = schema.select_by_tag(Tags.ITEM_ID).column_names[0]

    loader = Loader(sequence_testing_data, batch_size=8, shuffle=False)

    model = ml.Model(
        ml.InputBlockV2(
            schema,
            categorical=ml.Embeddings(
                schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner="mean"
            ),
            continuous=ml.Continuous(post=SequenceAggregator("mean")),
        ),
        ml.MLPBlock([32]),
        ml.CategoricalOutput(
            schema.select_by_name(target), default_loss="categorical_crossentropy"
        ),
    )

    predict_last = ml.SequencePredictLast(schema=schema.select_by_tag(Tags.SEQUENCE), target=target)

    testing_utils.model_test(
        model, loader, run_eagerly=run_eagerly, reload_model=True, fit_kwargs={"pre": predict_last}
    )

    metrics = model.evaluate(loader, steps=1, return_dict=True, pre=predict_last)
    assert len(metrics) > 0

    predictions = model.predict(loader, steps=1)
    assert predictions.shape == (8, 51997)


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


@pytest.mark.parametrize(
    "arguments", [([32], "relu"), ([64, 32], "relu"), ([32, 16], ["relu", "linear"])]
)
def test_mlp_block_dense_layer_activation(arguments):
    mlp = ml.MLPBlock(dimensions=arguments[0], activation=arguments[1])

    for idx, layer in enumerate(mlp.layers):
        activation_idx = arguments[1] if isinstance(arguments[1], str) else arguments[1][idx]
        assert layer.dense.activation.__name__ == activation_idx


def test_mlp_block_activation_dimensions_length_mismatch():
    with pytest.raises(ValueError) as excinfo:
        _ = ml.MLPBlock(dimensions=[32], activation=["relu", "linear"])
    assert "Activation and Dimensions length mismatch." in str(excinfo.value)


@pytest.mark.parametrize("low_rank_dim", [None, 32])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dropout", [0.5, None])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("normalization", [None, tf.keras.layers.BatchNormalization()])
def test_dense_residual_block(
    testing_data: Dataset,
    low_rank_dim,
    depth,
    use_bias,
    normalization,
    dropout,
    activation="selu",
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=regularizers.l2(1e-5),
    bias_regularizer=regularizers.l2(1e-5),
):
    inputs = ml.InputBlockV2(testing_data.schema)

    residual_block = ml.DenseResidualBlock(
        low_rank_dim=low_rank_dim,
        activation=activation,
        use_bias=use_bias,
        dropout=dropout,
        normalization=normalization,
        depth=depth,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    )

    batch = ml.sample_batch(testing_data, batch_size=100, include_targets=False)
    outputs = inputs(batch)
    input_dim = outputs.shape[-1]
    outputs = residual_block(outputs)

    assert list(outputs.shape) == [100, input_dim]

    res_block = residual_block
    if depth > 1:
        # Checks properties of the last stacked ResidualBlock
        res_block = residual_block.layers[-1]

    assert res_block.aggregation.activation.__name__ == activation
    assert res_block.layers[0].layers[0].dense.units == input_dim
    if low_rank_dim is not None:
        assert res_block.layers[0].layers[0].dense_u.units == low_rank_dim
    else:
        assert res_block.layers[0].layers[0].dense_u is None

    if dropout:
        assert res_block.layers[0].layers[1].rate == dropout
    if normalization:
        if normalization == "batch_norm":
            normalization = tf.keras.layers.BatchNormalization()

        assert res_block.layers[0].layers[-1].__class__.__name__ == normalization.__class__.__name__


@pytest.mark.parametrize("run_eagerly", [False, True])
def test_model_with_dense_residual(ecommerce_data: Dataset, run_eagerly: bool):
    model = ml.Model.from_block(
        ml.DenseResidualBlock(
            low_rank_dim=32,
            activation="relu",
            use_bias=True,
            dropout=0.5,
            normalization=tf.keras.layers.BatchNormalization(),
            depth=2,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=regularizers.l2(1e-5),
            bias_regularizer=regularizers.l2(1e-5),
        ),
        ecommerce_data.schema,
    )

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly, reload_model=False)
