import tensorflow as tf

from merlin.models.tf.data_augmentation.misc import ContinuousPowers


def test_continuous_powers():
    NUM_ROWS = 100

    inputs = {
        "cont_feat_1": tf.random.uniform((NUM_ROWS,)),
        "cont_feat_2": tf.random.uniform((NUM_ROWS,)),
    }

    powers = ContinuousPowers()

    outputs = powers(inputs)

    assert len(outputs) == len(inputs) * 3
    for key in inputs:
        assert key in outputs
        assert key + "_sqrt" in outputs
        assert key + "_pow" in outputs

        tf.assert_equal(tf.sqrt(inputs[key]), outputs[key + "_sqrt"])
        tf.assert_equal(tf.pow(inputs[key], 2), outputs[key + "_pow"])
