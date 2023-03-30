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
from typing import Any, Tuple, Union

import numpy as np
import pytest
import tensorflow as tf
from keras.utils import tf_inspect
from tensorflow.python.framework.test_util import disable_cudnn_autotune

import merlin.io
from merlin.models.tf.loader import Loader, sample_batch
from merlin.models.tf.models.base import Model
from merlin.models.tf.transforms.features import PrepareFeatures, expected_input_cols_from_schema
from merlin.schema import Tags


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
    dataset_or_loader: Union[merlin.io.Dataset, Loader],
    run_eagerly: bool = True,
    optimizer="adam",
    epochs: int = 1,
    reload_model: bool = False,
    fit_kwargs=None,
    **kwargs,
) -> Tuple[Model, Any]:
    """Generic model test. It will compile & fit the model and make sure it can be re-trained."""

    model.compile(run_eagerly=run_eagerly, optimizer=optimizer, **kwargs)

    if isinstance(dataset_or_loader, merlin.io.Dataset):
        dataloader = Loader(dataset_or_loader, batch_size=50)
    else:
        dataloader = dataset_or_loader

    fit_kwargs = fit_kwargs or {}
    losses = model.fit(dataloader, epochs=epochs, steps_per_epoch=1, **fit_kwargs)

    if reload_model:
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded_model = tf.keras.models.load_model(tmpdir)

        assert isinstance(loaded_model, type(model))

        x, y = sample_batch(dataloader, batch_size=50, prepare_features=False)
        batch = [(x, y)]

        model_preds = model.predict(iter(batch))
        loaded_model_preds = loaded_model.predict(iter(batch))

        if isinstance(model_preds, dict):
            for task_name in model_preds:
                tf.debugging.assert_near(
                    model_preds[task_name], loaded_model_preds[task_name], atol=1e-5
                )
        else:
            tf.debugging.assert_near(model_preds, loaded_model_preds, atol=1e-5)

        loaded_model.compile(run_eagerly=run_eagerly, optimizer=optimizer, **kwargs)
        loaded_model.fit(iter(batch))

        if model.input_schema:
            signature = loaded_model.signatures["serving_default"]
            signature_input_names = set(signature.structured_input_signature[1].keys())

            model_input_names = expected_input_cols_from_schema(model.input_schema)

            assert signature_input_names == set(model_input_names)

        return loaded_model, losses

    dataloader.stop()

    assert isinstance(model.from_config(model.get_config()), type(model))

    return model, losses


def test_model_signature(model, input_names, output_names):
    """Test that the model signature is correct."""

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
# and our CI tests multiple versions of tensorflow/keras
@disable_cudnn_autotune
def layer_test(
    layer_cls,
    args=None,
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
    args = args or []
    kwargs = kwargs or {}
    layer = layer_cls(*args, **kwargs)

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
        layer = layer_cls(*args, **kwargs)

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


def assert_allclose_according_to_type(
    a,
    b,
    rtol=1e-6,
    atol=1e-6,
    float_rtol=1e-6,
    float_atol=1e-6,
    half_rtol=1e-3,
    half_atol=1e-3,
    bfloat16_rtol=1e-2,
    bfloat16_atol=1e-2,
):
    """
    Similar to tf.test.TestCase.assertAllCloseAccordingToType()
    but this doesn't need a subclassing to run.
    """
    a = np.array(a)
    b = np.array(b)
    # types with lower tol are put later to overwrite previous ones.
    if (
        a.dtype == np.float32
        or b.dtype == np.float32
        or a.dtype == np.complex64
        or b.dtype == np.complex64
    ):
        rtol = max(rtol, float_rtol)
        atol = max(atol, float_atol)
    if a.dtype == np.float16 or b.dtype == np.float16:
        rtol = max(rtol, half_rtol)
        atol = max(atol, half_atol)
    if a.dtype == tf.bfloat16.as_numpy_dtype or b.dtype == tf.bfloat16.as_numpy_dtype:
        rtol = max(rtol, bfloat16_rtol)
        atol = max(atol, bfloat16_atol)

    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def assert_output_shape(output, expected_output_shape):
    def _get_shape(tensor_or_shape) -> tf.TensorShape:
        if hasattr(tensor_or_shape, "shape"):
            output_shape = tensor_or_shape.shape
        else:
            output_shape = tensor_or_shape
        return output_shape

    if isinstance(expected_output_shape, dict):
        for key in expected_output_shape.keys():
            output_shape = _get_shape(output[key])
            assert list(output_shape) == list(expected_output_shape[key])
    else:
        output_shape = _get_shape(output)
        assert list(output_shape) == list(expected_output_shape)


def loader_for_last_item_prediction(sequence_testing_data: merlin.io.Dataset, to_one_hot=True):
    schema = sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    prepare_features = PrepareFeatures(schema)

    class LastInteractionAsTarget:
        def compute_output_schema(self, input_schema):
            return input_schema

        @tf.function
        def __call__(self, inputs, targets=None):
            inputs = prepare_features(inputs)

            seq_item_id_col = schema.select_by_tag(Tags.ITEM_ID).column_names[0]
            targets = tf.squeeze(inputs[seq_item_id_col][:, -1:].flat_values, -1)
            if to_one_hot:
                targets = tf.one_hot(targets, schema[seq_item_id_col].int_domain.max + 1)

            for name in schema.select_by_tag(Tags.SEQUENCE).column_names:
                inputs[name] = inputs[name][:, :-1]

            col_names = list(inputs.keys())
            for k in col_names:
                if isinstance(inputs[k], tf.RaggedTensor):
                    inputs[f"{k}__values"] = inputs[k].values
                    inputs[f"{k}__offsets"] = inputs[k].row_splits
                    del inputs[k]

            return inputs, targets

    sequence_testing_data.schema = schema
    dataloader = Loader(sequence_testing_data, batch_size=50)
    _last_interaction_as_target = LastInteractionAsTarget()
    dataloader = dataloader.map(_last_interaction_as_target)
    return dataloader, schema
