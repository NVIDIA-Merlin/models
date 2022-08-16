import numpy as np
import pytest
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


def test_extract_topk():
    labels = tf.convert_to_tensor(
        [
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        ],
        tf.float32,
    )
    predictions = tf.convert_to_tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        tf.float32,
    )

    predictions_expected = tf.convert_to_tensor(
        [[10, 9, 8, 7, 6], [10, 9, 8, 7, 6], [1, 1, 1, 1, 1]], tf.float32
    )
    labels_expected = tf.convert_to_tensor(
        [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 1]],
        tf.float32,
    )
    label_relevant_counts_expected = tf.convert_to_tensor([3, 2, 3], tf.float32)

    cutoff = 5
    predictions_sorted, labels_sorted, label_relevant_counts = tf_utils.extract_topk(
        cutoff, predictions, labels
    )
    tf.debugging.assert_equal(label_relevant_counts_expected, label_relevant_counts)
    tf.debugging.assert_equal(labels_expected, labels_sorted[:, :cutoff])
    tf.debugging.assert_equal(predictions_expected, predictions_sorted[:, :cutoff])


@pytest.mark.parametrize("shuffle_ties", [True, False])
def test_extract_topk_shuffle_ties(shuffle_ties):
    labels = tf.convert_to_tensor(
        [
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
        tf.float32,
    )
    predictions = tf.convert_to_tensor([[10.0] * 20, [5.0] * 20, [6] * 20], tf.float32)
    label_relevant_counts_expected = tf.convert_to_tensor([5, 4, 4], tf.float32)
    cutoff = 10
    predictions_sorted, labels_sorted, label_relevant_counts = tf_utils.extract_topk(
        cutoff, predictions, labels, shuffle_ties=shuffle_ties
    )
    tf.debugging.assert_equal(label_relevant_counts, label_relevant_counts_expected)
    if shuffle_ties:
        assert not tf.reduce_all(tf.math.equal(labels_sorted, labels[:, :cutoff]))
    else:
        tf.debugging.assert_equal(labels_sorted, labels[:, :cutoff])
