import tensorflow as tf

import merlin.models.tf as mm


def test_expand_dims_same_axis():
    NUM_ROWS = 100

    # Creating some input sequences with padding in the end
    # (to emulate sessions with different lengths)
    inputs = {
        "cont_feat": tf.random.uniform((NUM_ROWS,)),
        "multi_hot_categ_feat": tf.random.uniform(
            (NUM_ROWS, 4), minval=1, maxval=100, dtype=tf.int32
        ),
    }

    expand_dims_op = mm.ExpandDims(expand_dims=-1)
    expanded_inputs = expand_dims_op(inputs)

    assert inputs.keys() == expanded_inputs.keys()
    assert list(expanded_inputs["cont_feat"].shape) == [NUM_ROWS, 1]
    assert list(expanded_inputs["multi_hot_categ_feat"].shape) == [NUM_ROWS, 4, 1]


def test_expand_dims_axis_as_dict():
    NUM_ROWS = 100

    # Creating some input sequences with padding in the end
    # (to emulate sessions with different lengths)
    inputs = {
        "cont_feat1": tf.random.uniform((NUM_ROWS,)),
        "cont_feat2": tf.random.uniform((NUM_ROWS,)),
        "multi_hot_categ_feat": tf.random.uniform(
            (NUM_ROWS, 4), minval=1, maxval=100, dtype=tf.int32
        ),
    }

    expand_dims_op = mm.ExpandDims(expand_dims={"cont_feat2": 0, "multi_hot_categ_feat": 1})
    expanded_inputs = expand_dims_op(inputs)

    assert inputs.keys() == expanded_inputs.keys()

    assert list(expanded_inputs["cont_feat1"].shape) == [NUM_ROWS]
    assert list(expanded_inputs["cont_feat2"].shape) == [1, NUM_ROWS]
    assert list(expanded_inputs["multi_hot_categ_feat"].shape) == [NUM_ROWS, 1, 4]
