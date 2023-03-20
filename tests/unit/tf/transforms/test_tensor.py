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
import tensorflow as tf

import merlin.models.tf as mm
from merlin.schema import ColumnSchema, Schema


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


def test_list_to_dense():
    NUM_ROWS = 100
    MAX_LEN = 10
    inputs = {
        "cont_feat": tf.random.uniform((NUM_ROWS,)),
        "multi_hot_categ_feat": tf.RaggedTensor.from_tensor(
            tf.random.uniform((NUM_ROWS, 4), minval=1, maxval=100, dtype=tf.int32)
        ),
        "multi_hot_embedding_feat": tf.RaggedTensor.from_tensor(
            tf.random.uniform((NUM_ROWS, 4, 32), minval=1, maxval=100, dtype=tf.int32)
        ),
    }

    schema = Schema(
        [
            ColumnSchema("cont_feat", dtype="float32").with_shape((NUM_ROWS,)),
            ColumnSchema("multi_hot_categ_feat", dtype="int32").with_shape((NUM_ROWS, MAX_LEN)),
            ColumnSchema("multi_hot_embedding_feat", dtype="int32").with_shape(
                (NUM_ROWS, MAX_LEN, 32)
            ),
        ]
    )

    list_to_dense_op = mm.ToDense(schema)
    dense_inputs = list_to_dense_op(inputs)
    output_shapes = list_to_dense_op.compute_output_shape({k: v.shape for k, v in inputs.items()})

    assert inputs.keys() == dense_inputs.keys()
    assert list(dense_inputs["cont_feat"].shape) == [NUM_ROWS]
    assert list(dense_inputs["multi_hot_categ_feat"].shape) == [NUM_ROWS, MAX_LEN]
    assert list(dense_inputs["multi_hot_embedding_feat"].shape) == [NUM_ROWS, MAX_LEN, 32]
    assert list(output_shapes["multi_hot_embedding_feat"]) == [NUM_ROWS, MAX_LEN, 32]
