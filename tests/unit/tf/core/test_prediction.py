from typing import Dict

import numpy as np
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.core.prediction import Prediction


# TODO: Fix this in graph-mode
@pytest.mark.parametrize("run_eagerly", [True])
def test_layer_with_features_in_model(music_streaming_data: Dataset, run_eagerly):
    class Dummy(tf.keras.layers.Layer):
        @tf.function
        def call(self, inputs, features):
            outputs = {}
            for key, val in inputs.items():
                if isinstance(val, tuple):
                    outputs[key] = val
                else:
                    outputs[key] = features[key]

            return outputs

        def compute_output_shape(self, input_shape):
            return input_shape

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
        def call(self, inputs: Dict[str, tf.Tensor], targets=None):
            if targets is not None:
                if isinstance(targets, dict):
                    flipped = {}
                    for key in targets:
                        flipped[key] = self.flip_target(targets[key])
                else:
                    flipped = self.flip_target(targets)

                return Prediction(inputs, flipped)

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
    outputs, context = model(features, targets=targets, training=True, output_context=True)

    flipped = np.logical_not(targets["click"].numpy()).astype(np.int)
    assert np.array_equal(context.targets["click"], flipped)
