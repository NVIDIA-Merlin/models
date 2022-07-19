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
import pathlib
import platform
import tempfile
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
import tensorflow as tf
from keras.utils import tf_inspect
from tensorflow.python.framework.test_util import disable_cudnn_autotune

import merlin.io
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.models.base import Model
from merlin.schema import Schema


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
    dataset: Union[merlin.io.Dataset, BatchedDataset],
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


def get_model_inputs(schema: Schema, list_cols: Optional[Sequence[str]] = None):
    list_cols = list_cols or []
    features = schema.column_names

    # Right now the model expects a tuple for each list-column
    for list_col in list_cols:
        features.pop(features.index(list_col))
        for i in ["1", "2"]:
            features.append(f"{list_col}_{i}")

    return features


def test_model_signature(model, input_names, output_names):
    signatures = getattr(model, "signatures", {}) or {}
    default_signature = signatures.get("serving_default")

    if not default_signature:
        # roundtrip saved self.model to disk to generate signature if it doesn't exist

        with tempfile.TemporaryDirectory() as tmp_dir:
            tf_model_path = pathlib.Path(tmp_dir) / "model.savedmodel"
            model.save(tf_model_path, include_optimizer=False)
            reloaded = tf.keras.models.load_model(tf_model_path)
            default_signature = reloaded.signatures["serving_default"]

    model_inputs = set(default_signature.structured_input_signature[1].keys())
    assert set(input_names) == model_inputs

    model_outputs = set(default_signature.structured_outputs.keys())
    assert set(output_names) == model_outputs


def string_test(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def numeric_test(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)


# This function is copied from keras/testing_infra/test_utils.py
# We need it here because this was not publicly exposed prior to 2.9.0
# and our CI tests muliple versions of tensorflow/keras
@disable_cudnn_autotune
def layer_test(
    layer_cls,
    kwargs=None,
    input_shape=None,
    input_dtype=None,
    input_data=None,
    expected_output=None,
    expected_output_dtype=None,
    expected_output_shape=None,
    validate_training=True,
    adapt_data=None,
    custom_objects=None,
    test_harness=None,
    supports_masking=None,
):
    """Test routine for a layer with a single input and single output.
    Args:
      layer_cls: Layer class object.
      kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
      input_shape: Input shape tuple.
      input_dtype: Data type of the input data.
      input_data: Numpy array of input data.
      expected_output: Numpy array of the expected output.
      expected_output_dtype: Data type expected for the output.
      expected_output_shape: Shape tuple for the expected shape of the output.
      validate_training: Whether to attempt to validate training on this layer.
        This might be set to False for non-differentiable layers that output
        string or integer values.
      adapt_data: Optional data for an 'adapt' call. If None, adapt() will not
        be tested for this layer. This is only relevant for PreprocessingLayers.
      custom_objects: Optional dictionary mapping name strings to custom objects
        in the layer class. This is helpful for testing custom layers.
      test_harness: The Tensorflow test, if any, that this function is being
        called in.
      supports_masking: Optional boolean to check the `supports_masking`
        property of the layer. If None, the check will not be performed.
    Returns:
      The output data (Numpy array) returned by the layer, for additional
      checks to be done by the calling code.
    Raises:
      ValueError: if `input_shape is None`.
    """
    if input_data is None:
        if input_shape is None:
            raise ValueError("input_shape is None")
        if not input_dtype:
            input_dtype = "float32"
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        input_data = 10 * np.random.random(input_data_shape)
        if input_dtype[:5] == "float":
            input_data -= 0.5
        input_data = input_data.astype(input_dtype)
    elif input_shape is None:
        input_shape = input_data.shape
    if input_dtype is None:
        input_dtype = input_data.dtype
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    if tf.as_dtype(expected_output_dtype) == tf.string:
        if test_harness:
            assert_equal = test_harness.assertAllEqual
        else:
            assert_equal = string_test
    else:
        if test_harness:
            assert_equal = test_harness.assertAllClose
        else:
            assert_equal = numeric_test

    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    if supports_masking is not None and layer.supports_masking != supports_masking:
        raise AssertionError(
            "When testing layer %s, the `supports_masking` property is %r"
            "but expected to be %r.\nFull kwargs: %s"
            % (
                layer_cls.__name__,
                layer.supports_masking,
                supports_masking,
                kwargs,
            )
        )

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if "weights" in tf_inspect.getargspec(layer_cls.__init__):
        kwargs["weights"] = weights
        layer = layer_cls(**kwargs)

    # test in functional API
    x = tf.keras.layers.Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    if tf.keras.backend.dtype(y) != expected_output_dtype:
        raise AssertionError(
            "When testing layer %s, for input %s, found output "
            "dtype=%s but expected to find %s.\nFull kwargs: %s"
            % (
                layer_cls.__name__,
                x,
                tf.keras.backend.dtype(y),
                expected_output_dtype,
                kwargs,
            )
        )

    def assert_shapes_equal(expected, actual):
        """Asserts that the output shape from the layer matches the actual
        shape."""
        if len(expected) != len(actual):
            raise AssertionError(
                "When testing layer %s, for input %s, found output_shape="
                "%s but expected to find %s.\nFull kwargs: %s"
                % (layer_cls.__name__, x, actual, expected, kwargs)
            )

        for expected_dim, actual_dim in zip(expected, actual):
            if isinstance(expected_dim, tf.compat.v1.Dimension):
                expected_dim = expected_dim.value
            if isinstance(actual_dim, tf.compat.v1.Dimension):
                actual_dim = actual_dim.value
            if expected_dim is not None and expected_dim != actual_dim:
                raise AssertionError(
                    "When testing layer %s, for input %s, found output_shape="
                    "%s but expected to find %s.\nFull kwargs: %s"
                    % (layer_cls.__name__, x, actual, expected, kwargs)
                )

    if expected_output_shape is not None:
        assert_shapes_equal(tf.TensorShape(expected_output_shape), y.shape)

    # check shape inference
    model = tf.keras.models.Model(x, y)
    computed_output_shape = tuple(layer.compute_output_shape(tf.TensorShape(input_shape)).as_list())
    computed_output_signature = layer.compute_output_signature(
        tf.TensorSpec(shape=input_shape, dtype=input_dtype)
    )
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert_shapes_equal(computed_output_shape, actual_output_shape)
    assert_shapes_equal(computed_output_signature.shape, actual_output_shape)
    if computed_output_signature.dtype != actual_output.dtype:
        raise AssertionError(
            "When testing layer %s, for input %s, found output_dtype="
            "%s but expected to find %s.\nFull kwargs: %s"
            % (
                layer_cls.__name__,
                x,
                actual_output.dtype,
                computed_output_signature.dtype,
                kwargs,
            )
        )
    if expected_output is not None:
        assert_equal(actual_output, expected_output)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = tf.keras.models.Model.from_config(model_config, custom_objects)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        assert_equal(output, actual_output)

    # test training mode (e.g. useful for dropout tests)
    # Rebuild the model to avoid the graph being reused between predict() and
    # See b/120160788 for more details. This should be mitigated after 2.0.
    layer_weights = layer.get_weights()  # Get the layer weights BEFORE training.
    if validate_training:
        model = tf.keras.models.Model(x, layer(x))
        model.compile("rmsprop", "mse", weighted_metrics=["acc"])
        model.train_on_batch(input_data, actual_output)

    # test as first layer in Sequential API
    layer_config = layer.get_config()
    layer_config["batch_input_shape"] = input_shape
    layer = layer.__class__.from_config(layer_config)

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape[1:], dtype=input_dtype))
    model.add(layer)

    layer.set_weights(layer_weights)
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(computed_output_shape, actual_output_shape):
        if expected_dim is not None:
            if expected_dim != actual_dim:
                raise AssertionError(
                    "When testing layer %s **after deserialization**, "
                    "for input %s, found output_shape="
                    "%s but expected to find inferred shape %s.\n"
                    "Full kwargs: %s"
                    % (
                        layer_cls.__name__,
                        x,
                        actual_output_shape,
                        computed_output_shape,
                        kwargs,
                    )
                )
    if expected_output is not None:
        assert_equal(actual_output, expected_output)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = tf.keras.models.Sequential.from_config(model_config, custom_objects)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        assert_equal(output, actual_output)

    # for further checks in the caller function
    return actual_output
