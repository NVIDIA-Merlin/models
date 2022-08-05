import numpy as np
import tensorflow as tf

from merlin.models.tf.utils import tf_utils


class TestTensorInitializer:
    def test_serialize(self):
        shape = (10, 1)
        initializer = tf_utils.TensorInitializer(np.random.rand(*shape).astype(np.float32))
        variable = tf.keras.backend.variable(initializer(shape))
        output = tf.keras.backend.get_value(variable)

        config = tf.keras.initializers.serialize(initializer)
        reloaded_initializer = tf.keras.initializers.deserialize(config)

        variable = tf.keras.backend.variable(reloaded_initializer(shape))
        output_2 = tf.keras.backend.get_value(variable)

        np.testing.assert_array_almost_equal(output, output_2)

    def test_model_load(self, tmpdir):
        shape = (10, 1)
        initializer = tf_utils.TensorInitializer(np.random.rand(*shape))

        inputs = tf.keras.Input((10,))
        outputs = tf.keras.layers.Dense(1, kernel_initializer=initializer)(inputs)
        model = tf.keras.models.Model(inputs, outputs)

        input_tensor = tf.constant(np.random.rand(1, 10))

        output = model(input_tensor).numpy()

        model.save(tmpdir)
        reloaded_model = tf.keras.models.load_model(tmpdir)
        output_2 = reloaded_model(input_tensor).numpy()

        np.testing.assert_array_almost_equal(output, output_2)
