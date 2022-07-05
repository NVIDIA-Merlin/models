from typing import Dict

import numpy as np
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.blocks.core.context import Context, ContextTensor, ContextTensorShape

# tf.config.run_functions_eagerly(True)


def test_context(music_streaming_data: Dataset):
    schema = music_streaming_data.schema
    features, targets = mm.sample_batch(music_streaming_data, batch_size=10, to_dense=True)

    context = Context(schema, features, targets)
    context_1 = Context(schema, features, targets)

    assert context == context_1

    context_tensor = ContextTensor(features["user_id"], context)
    context_tensor_1 = ContextTensor(features["user_id"], context_1)

    assert context_tensor == context_tensor_1


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_context_in_model(music_streaming_data: Dataset, run_eagerly):
    class Dummy(tf.keras.layers.Layer):
        _USES_CONTEXT = True

        @tf.function
        # @uses_context
        def call(self, inputs: ContextTensor[Dict[str, tf.Tensor]]):
            outputs = {}
            for key, val in inputs.value.items():
                if isinstance(val, tuple):
                    outputs[key] = val
                else:
                    outputs[key] = inputs.context.features[key]

            return outputs

        def compute_output_shape(self, input_shape: ContextTensorShape):
            return input_shape.value

    model = mm.Model(
        Dummy(),
        mm.InputBlock(music_streaming_data.schema),
        mm.MLPBlock([64]),
        mm.BinaryClassificationTask("click"),
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    model.fit(music_streaming_data, epochs=1, batch_size=10)


@pytest.mark.parametrize("run_eagerly", [True])
def test_model_pre_transforming_targets(ecommerce_data: Dataset, run_eagerly):
    class FlipTargets(tf.keras.layers.Layer):
        _USES_CONTEXT = True

        def call(self, inputs: ContextTensor[Dict[str, tf.Tensor]]):
            _targets = inputs.context.targets
            if _targets:
                if isinstance(_targets, dict):
                    targets = {}
                    for key in _targets:
                        targets[key] = self.flip_target(_targets[key])
                else:
                    targets = self.flip_target(_targets)

                return ContextTensor(inputs.value, inputs.context.with_targets(targets))

            return inputs

        @staticmethod
        def flip_target(target):
            dtype = target.dtype

            return tf.cast(tf.math.logical_not(tf.cast(target, tf.bool)), dtype)

    model = mm.Model(
        mm.InputBlock(ecommerce_data.schema),
        mm.MLPBlock([64]),
        mm.BinaryClassificationTask("click"),
        pre=FlipTargets(),
    )

    features, targets = mm.sample_batch(ecommerce_data, batch_size=100)
    outputs = model(features, targets=targets)

    flipped = np.logical_not(targets["click"].numpy()).astype(np.int)
    assert np.array_equal(outputs.context.targets["click"], flipped)
