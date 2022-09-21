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
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.test import TestCase

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.transforms.features import ContinuousPowers
from merlin.models.tf.utils import testing_utils
from merlin.models.utils.schema_utils import create_categorical_column
from merlin.schema import Schema, Tags


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_categorical_encoding_as_pre(ecommerce_data: Dataset, run_eagerly):
    schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    body = mm.ParallelBlock(
        mm.TabularBlock.from_schema(schema=schema, pre=mm.CategoryEncoding(schema)),
        is_input=True,
    ).connect(mm.MLPBlock([32]))
    model = mm.Model(body, mm.BinaryClassificationTask("click"))

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_categorical_encoding_in_model(ecommerce_data: Dataset, run_eagerly):
    schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    branches = {
        "one_hot": mm.CategoryEncoding(schema, is_input=True),
        "features": mm.InputBlock(ecommerce_data.schema),
    }
    body = mm.ParallelBlock(branches, is_input=True).connect(mm.MLPBlock([32]))
    model = mm.Model(body, mm.BinaryClassificationTask("click"))

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


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


def test_hashedcross_int():
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=3),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=3),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant(["A", "B"])
    inputs["cat2"] = tf.constant([101, 102])
    hashed_cross_op = mm.HashedCross(schema=schema, num_bins=10, output_mode="int")
    outputs = hashed_cross_op(inputs)
    output_name, output_value = outputs.popitem()

    assert output_name == "cross_cat1_cat2"
    assert output_value.shape.as_list() == [2, 1]
    test_case.assertAllClose(output_value, [[1], [6]])


def test_hashedcross_1d():
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant(["A", "B", "A", "B", "A"])
    inputs["cat2"] = tf.constant([101, 101, 101, 102, 102])
    hashed_cross_op = mm.HashedCross(schema=schema, num_bins=10, output_mode="int")
    outputs = hashed_cross_op(inputs)
    _, output_value = outputs.popitem()

    assert output_value.shape.as_list() == [5, 1]
    test_case.assertAllClose(output_value, [[1], [4], [1], [6], [3]])


def test_hashedcross_2d():
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
    inputs["cat2"] = tf.constant([[101], [101], [101], [102], [102]])
    hashed_cross_op = mm.HashedCross(schema=schema, num_bins=10, output_mode="int")
    outputs = hashed_cross_op(inputs)
    _, output_value = outputs.popitem()

    assert output_value.shape.as_list() == [5, 1]
    test_case.assertAllClose(output_value, [[1], [4], [1], [6], [3]])


def test_hashedcross_output_shape():
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
        ]
    )
    inputs_shape = {}
    inputs_shape["cat1"] = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]]).shape
    inputs_shape["cat2"] = tf.constant([[101], [101], [101], [102], [102]]).shape
    hashed_cross = mm.HashedCross(schema=schema, num_bins=10, output_mode="int")
    outputs = hashed_cross.compute_output_shape(inputs_shape)
    _, output_shape = outputs.popitem()

    assert output_shape == [5, 1]


def test_hashedcross_output_shape_one_hot():
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
        ]
    )
    inputs_shape = {}
    inputs_shape["cat1"] = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]]).shape
    inputs_shape["cat2"] = tf.constant([[101], [101], [101], [102], [102]]).shape
    output_name = "cross_out"
    hashed_cross = mm.HashedCross(
        schema=schema, num_bins=10, output_mode="one_hot", output_name=output_name
    )
    outputs = hashed_cross.compute_output_shape(inputs_shape)
    _output_name, output_shape = outputs.popitem()

    assert output_shape == [5, 10]
    assert _output_name == output_name


def test_hashedcross_less_bins():
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant([["A"], ["B"], ["C"], ["D"], ["A"], ["B"], ["A"]])
    inputs["cat2"] = tf.constant([[101], [102], [101], [101], [101], [102], [103]])
    hashed_cross_op = mm.HashedCross(schema=schema, num_bins=4, output_mode="one_hot", sparse=True)
    outputs = hashed_cross_op(inputs)
    _, output_value = outputs.popitem()
    output_value = tf.sparse.to_dense(output_value)

    assert output_value.shape.as_list() == [7, 4]


def test_hashedcross_output_mode():
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant([["A"], ["B"], ["C"], ["D"], ["A"], ["B"], ["A"]])
    inputs["cat2"] = tf.constant([[101], [102], [101], [101], [101], [102], [103]])

    hashed_cross_op = mm.HashedCross(schema=schema, num_bins=4, output_mode="one_hot", sparse=True)
    outputs = hashed_cross_op(inputs)
    _, output_value = outputs.popitem()
    assert isinstance(output_value, tf.SparseTensor) is True
    assert output_value.shape.as_list() == [7, 4]

    hashed_cross_op = mm.HashedCross(schema=schema, num_bins=4, output_mode="one_hot", sparse=False)
    outputs = hashed_cross_op(inputs)
    _, output_value = outputs.popitem()
    assert isinstance(output_value, tf.Tensor) is True
    assert output_value.shape.as_list() == [7, 4]


def test_hashedcross_onehot_output():
    test_case = TestCase()

    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
    inputs["cat2"] = tf.constant([[101], [101], [101], [102], [102]])
    hashed_cross_op = mm.HashedCross(schema=schema, num_bins=5, output_mode="one_hot", sparse=True)
    outputs = hashed_cross_op(inputs)
    _, output_value = outputs.popitem()
    output_value = tf.sparse.to_dense(output_value)

    assert output_value.shape.as_list() == [5, 5]
    test_case.assertAllClose(
        output_value,
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
    )


def test_hashedcross_single_input_fails():
    test_case = TestCase()
    schema = Schema([create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20)])
    with test_case.assertRaisesRegex(ValueError, "at least two features"):
        mm.HashedCross(num_bins=10, schema=schema, output_mode="int")([tf.constant(1)])


def test_hashedcross_from_config():
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
    inputs["cat2"] = tf.constant([[101], [101], [101], [102], [102]])
    hashed_cross_op = mm.HashedCross(schema=schema, num_bins=5, output_mode="one_hot", sparse=False)
    cloned_hashed_cross_op = mm.HashedCross.from_config(hashed_cross_op.get_config())
    original_outputs = hashed_cross_op(inputs)
    cloned_outputs = cloned_hashed_cross_op(inputs)
    _, original_output_value = original_outputs.popitem()
    _, cloned_output_value = cloned_outputs.popitem()

    test_case.assertAllEqual(cloned_output_value, original_output_value)


def test_hashedcrosses_in_parallelblock():
    test_case = TestCase()

    schema_0 = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
        ]
    )
    schema_1 = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=2),
            create_categorical_column("cat3", tags=[Tags.CATEGORICAL], num_items=3),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
    inputs["cat2"] = tf.constant([[101], [101], [101], [102], [102]])
    inputs["cat3"] = tf.constant([[1], [0], [1], [2], [2]])
    hashed_cross_0 = mm.HashedCross(
        schema=schema_0, num_bins=5, output_mode="one_hot", sparse=True, output_name="cross_0"
    )
    hashed_cross_1 = mm.HashedCross(
        schema=schema_1, num_bins=10, output_mode="one_hot", sparse=True, output_name="cross_1"
    )
    hashed_crosses = mm.ParallelBlock([hashed_cross_0, hashed_cross_1])
    outputs = hashed_crosses(inputs)
    output_value_0 = outputs["cross_0"]
    output_value_0 = tf.sparse.to_dense(output_value_0)

    assert output_value_0.shape.as_list() == [5, 5]
    test_case.assertAllClose(
        output_value_0,
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
    )
    output_value_1 = outputs["cross_1"]
    output_value_1 = tf.sparse.to_dense(output_value_1)

    assert output_value_1.shape.as_list() == [5, 10]
    test_case.assertAllClose(
        output_value_1,
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ],
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_hashedcross_as_pre(ecommerce_data: Dataset, run_eagerly):
    cross_schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    body = mm.ParallelBlock(
        mm.TabularBlock.from_schema(
            schema=cross_schema, pre=mm.HashedCross(cross_schema, num_bins=1000)
        ),
        is_input=True,
    ).connect(mm.MLPBlock([64]))
    model = mm.Model(body, mm.BinaryClassificationTask("click"))

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_hashedcross_in_model(ecommerce_data: Dataset, run_eagerly):
    cross_schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    branches = {
        "cross_product": mm.HashedCross(cross_schema, num_bins=1000, is_input=True),
        "features": mm.InputBlock(ecommerce_data.schema),
    }
    body = mm.ParallelBlock(branches, is_input=True).connect(mm.MLPBlock([64]))
    model = mm.Model(body, mm.BinaryClassificationTask("click"))

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    testing_utils.model_test(model, ecommerce_data)


def test_category_encoding_different_input_different_output():
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("dense_feature", tags=[Tags.CATEGORICAL], num_items=4),
            create_categorical_column("sparse_feature", tags=[Tags.CATEGORICAL], num_items=5),
        ]
    )
    inputs = {}
    inputs["dense_feature"] = tf.constant([[1, 2, 3], [3, 3, 0]])
    inputs["sparse_feature"] = tf.sparse.from_dense(
        np.array([[1, 2, 3, 0], [0, 3, 1, 0]], dtype=np.int64)
    )

    # 1. Sparse output
    category_encoding = mm.CategoryEncoding(
        schema=schema,
        output_mode="count",
        sparse=True,
    )
    outputs = category_encoding(inputs)

    # The expected output["dense_feature"] should be (X for missing value):
    # [[X, 1, 1, 1]
    #  [1, X, X, 2]]
    expected_indices_1 = [[0, 1], [0, 2], [0, 3], [1, 0], [1, 3]]
    expected_values_1 = [1, 1, 1, 1, 2]
    test_case.assertAllEqual(expected_values_1, outputs["dense_feature"].values)
    test_case.assertAllEqual(expected_indices_1, outputs["dense_feature"].indices)

    expected_indices_2 = [[0, 1], [0, 2], [0, 3], [1, 1], [1, 3]]
    expected_values_2 = [1, 1, 1, 1, 1]
    test_case.assertAllEqual(expected_values_2, outputs["sparse_feature"].values)
    test_case.assertAllEqual(expected_indices_2, outputs["sparse_feature"].indices)

    # 2. Dense output
    category_encoding = mm.CategoryEncoding(
        schema=schema,
        output_mode="count",
        sparse=False,
    )
    expected_1 = [[0, 1, 1, 1, 0], [1, 0, 0, 2, 0]]
    expected_2 = [[0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0]]
    outputs = category_encoding(inputs)
    test_case.assertAllEqual(expected_1, outputs["dense_feature"])
    test_case.assertAllEqual(expected_2, outputs["sparse_feature"])


def test_category_encoding_invalid_input():
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("ragged_feature", tags=[Tags.CATEGORICAL], num_items=5),
        ]
    )
    inputs = {}
    inputs["ragged_feature"] = tf.ragged.constant([[1, 2, 3], [3, 1], []])
    category_encoding = mm.CategoryEncoding(
        schema=schema,
        output_mode="count",
        sparse=False,
    )
    with test_case.assertRaisesRegex(ValueError, "inputs should not contain a RaggedTensor"):
        category_encoding(inputs)


@pytest.mark.parametrize("input", [np.array([[1, 2, 3, 4], [4, 3, 1, 4]])])
@pytest.mark.parametrize("weight", [np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]])])
def test_category_encoding_weightd_count_dense(input, weight):
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("feature", tags=[Tags.CATEGORICAL], num_items=5),
        ]
    )

    expected_output = [[0, 0.1, 0.2, 0.3, 0.4, 0], [0, 0.4, 0, 0.1, 0.5, 0]]
    # pyformat: enable
    expected_output_shape = [2, 6]

    category_encoding = mm.CategoryEncoding(
        schema=schema, output_mode="count", count_weights=weight
    )

    inputs = {}
    inputs["feature"] = input
    outputs = category_encoding(inputs)
    test_case.assertAllEqual(expected_output_shape, outputs["feature"].shape.as_list())
    test_case.assertAllClose(expected_output, outputs["feature"])


@pytest.mark.parametrize("input", [tf.sparse.from_dense(np.array([[1, 2, 3, 4], [4, 3, 1, 4]]))])
@pytest.mark.parametrize(
    "weight", [tf.sparse.from_dense(np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]]))]
)
def test_category_encoding_weightd_count_sparse(input, weight):
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("feature", tags=[Tags.CATEGORICAL], num_items=5),
        ]
    )

    expected_output = [[0, 0.1, 0.2, 0.3, 0.4, 0], [0, 0.4, 0, 0.1, 0.5, 0]]
    # pyformat: enable
    expected_output_shape = [2, 6]

    category_encoding = mm.CategoryEncoding(
        schema=schema, output_mode="count", count_weights=weight
    )

    inputs = {}
    inputs["feature"] = input
    outputs = category_encoding(inputs)
    test_case.assertAllEqual(expected_output_shape, outputs["feature"].shape.as_list())
    test_case.assertAllClose(expected_output, outputs["feature"])


@pytest.mark.parametrize("input", [tf.sparse.from_dense(np.array([[1, 2, 3, 4], [4, 3, 1, 4]]))])
@pytest.mark.parametrize("weight", [np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3]])])
def test_category_encoding_weightd_count_not_match(input, weight):
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("feature", tags=[Tags.CATEGORICAL], num_items=5),
        ]
    )
    category_encoding = mm.CategoryEncoding(
        schema=schema, output_mode="count", count_weights=weight
    )
    inputs = {}
    inputs["feature"] = input
    with test_case.assertRaisesRegex(
        ValueError, "Argument `weights` must be a SparseTensor if `values` is a SparseTensor"
    ):
        category_encoding(inputs)


@pytest.mark.parametrize(
    "input",
    [
        tf.convert_to_tensor([[1, 2, 3, 0], [0, 3, 1, 0]]),
        tf.sparse.from_dense(np.array([[1, 2, 3, 0], [0, 3, 1, 0]])),
    ],
)
def test_category_encoding_multi_hot_2d_input(input):
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("feature", tags=[Tags.CATEGORICAL], num_items=5),
        ]
    )

    if isinstance(input, tf.SparseTensor):
        expected_output = [[0, 1, 1, 1, 0, 0], [0, 1, 0, 1, 0, 0]]
    else:
        # Dense tensors with 0 will have it included in the multi-hot output
        expected_output = [[1, 1, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0]]
    # pyformat: enable
    expected_output_shape = [2, 6]

    category_encoding = mm.CategoryEncoding(schema=schema, output_mode="multi_hot")

    inputs = {}
    inputs["feature"] = input
    outputs = category_encoding(inputs)
    test_case.assertAllEqual(expected_output_shape, outputs["feature"].shape.as_list())
    test_case.assertAllClose(expected_output, outputs["feature"])


@pytest.mark.parametrize(
    "input",
    [
        tf.convert_to_tensor([1, 2, 0]),
        tf.sparse.from_dense([1, 2, 0]),
        tf.convert_to_tensor([[1], [2], [0]]),
        tf.sparse.from_dense([[1], [2], [0]]),
    ],
)
def test_category_encoding_multi_hot_single_value(input):
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("feature", tags=[Tags.CATEGORICAL], num_items=5),
        ]
    )

    if isinstance(input, tf.SparseTensor):
        expected_output = [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    else:
        # Dense tensors with 0 will have it included in the multi-hot output
        expected_output = [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
    # pyformat: enable
    expected_output_shape = [3, 6]

    category_encoding = mm.CategoryEncoding(schema=schema, output_mode="multi_hot")

    inputs = {}
    inputs["feature"] = input
    outputs = category_encoding(inputs)
    test_case.assertAllEqual(expected_output_shape, outputs["feature"].shape.as_list())
    test_case.assertAllClose(expected_output, outputs["feature"])


@pytest.mark.parametrize(
    "input",
    [
        tf.convert_to_tensor([1, 2, 3, 0]),
        tf.sparse.from_dense(np.array([1, 2, 3, 0])),
        tf.convert_to_tensor([[1], [2], [3], [0]]),
        tf.sparse.from_dense(np.array([[1], [2], [3], [0]])),
    ],
)
def test_category_encoding_one_hot(input):
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("feature", tags=[Tags.CATEGORICAL], num_items=5),
        ]
    )

    if isinstance(input, tf.SparseTensor):
        expected_output = [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    else:
        # Dense tensors with 0 will have it included in the multi-hot output
        expected_output = [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ]
    # pyformat: enable
    expected_output_shape = [4, 6]

    category_encoding = mm.CategoryEncoding(schema=schema, output_mode="one_hot")

    inputs = {}
    inputs["feature"] = input
    outputs = category_encoding(inputs)
    test_case.assertAllEqual(expected_output_shape, outputs["feature"].shape.as_list())
    test_case.assertAllClose(expected_output, outputs["feature"])


@pytest.mark.parametrize(
    "input",
    [tf.convert_to_tensor([[1, 2], [2, 0]]), tf.sparse.from_dense(np.array([[1, 2], [2, 0]]))],
)
def test_category_encoding_one_hot_2D_input_should_raise(input):
    test_case = TestCase()
    schema = Schema([create_categorical_column("feature", tags=[Tags.CATEGORICAL], num_items=5)])
    category_encoding = mm.CategoryEncoding(schema=schema, output_mode="one_hot")
    inputs = {}
    inputs["feature"] = input
    with test_case.assertRaisesRegex(
        ValueError, r"One-hot accepts input tensors that are squeezable to 1D"
    ):
        category_encoding(inputs)


@pytest.mark.parametrize(
    "input",
    [
        tf.convert_to_tensor([[[1], [2]], [[2], [0]]]),
        tf.sparse.from_dense(np.array([[[1], [2]], [[2], [0]]])),
    ],
)
def test_category_encoding_should_raise_if_input_3D(input):
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("feature", tags=[Tags.CATEGORICAL], num_items=5),
        ]
    )
    category_encoding = mm.CategoryEncoding(schema=schema, output_mode="multi_hot")
    inputs = {}
    inputs["feature"] = input
    with test_case.assertRaisesRegex(
        Exception, r"`CategoryEncoding` only accepts 1D or 2D-shaped inputs"
    ):
        category_encoding(inputs)


def test_hashedcrossall():
    schema = Schema(
        [
            # num_items: 0, 1, 2 thus cardinality = 3
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=2),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=2),
            create_categorical_column("cat3", tags=[Tags.CATEGORICAL], num_items=2),
            create_categorical_column("cat4", tags=[Tags.CATEGORICAL], num_items=3),
            create_categorical_column("cat5", tags=[Tags.CATEGORICAL], num_items=3),
            create_categorical_column("cat6", tags=[Tags.CATEGORICAL], num_items=3),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
    inputs["cat2"] = tf.constant([[101], [101], [101], [102], [102]])
    inputs["cat3"] = tf.constant([[1], [0], [1], [2], [2]])
    inputs["cat4"] = tf.constant([[1], [0], [1], [3], [2]])
    inputs["cat5"] = tf.constant([[1], [0], [1], [3], [2]])
    inputs["cat6"] = tf.constant([[1], [0], [1], [3], [2]])

    hashed_cross_all = mm.HashedCrossAll(
        schema=schema,
        infer_num_bins=True,
        output_mode="one_hot",
        sparse=True,
        max_num_bins=25,
        max_level=3,
        ignore_combinations=[["cat3", "cat4", "cat5"], ["cat1", "cat2"]],
    )

    outputs = hashed_cross_all(inputs)
    assert len(outputs) == 33

    output_value_0 = outputs["cross_cat1_cat3"]
    assert output_value_0.shape.as_list() == [5, 9]

    output_value_1 = outputs["cross_cat1_cat3_cat6"]
    assert output_value_1.shape.as_list() == [5, 25]


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_hashedcrossall_in_model(ecommerce_data: Dataset, run_eagerly):
    cross_schema = ecommerce_data.schema.select_by_name(
        names=["user_categories", "item_category", "item_brand"]
    )
    branches = {
        "cross_product": mm.HashedCrossAll(cross_schema, max_num_bins=1000, infer_num_bins=True),
        "features": mm.InputBlock(ecommerce_data.schema),
    }
    body = mm.ParallelBlock(branches, is_input=True).connect(mm.MLPBlock([64]))
    model = mm.Model(body, mm.BinaryClassificationTask("click"))

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize(
    "only_selected_in_schema",
    [False, True],
)
def test_to_dense(only_selected_in_schema):
    test_case = TestCase()

    if only_selected_in_schema:
        schema = Schema(
            [
                create_categorical_column("feature1", tags=[Tags.CATEGORICAL], num_items=5),
                create_categorical_column("feature2", tags=[Tags.CATEGORICAL], num_items=5),
            ]
        )
        to_dense = mm.ToDense(schema.select_by_name("feature1"))
    else:
        # Converts all features to dense
        to_dense = mm.ToDense()

    feature1_dense = np.array([[1, 2, 3, 0], [0, 3, 1, 0]])
    feature2_dense = np.array([[2, 3, 4, 0], [2, 0, 1, 0]])

    data = {
        "feature1": tf.sparse.from_dense(feature1_dense),
        "feature2": tf.sparse.from_dense(feature2_dense),
    }

    output = to_dense(data)

    assert output["feature1"].shape == data["feature1"].shape
    assert output["feature2"].shape == data["feature2"].shape

    assert isinstance(output["feature1"], tf.Tensor)
    if only_selected_in_schema:
        assert isinstance(output["feature2"], tf.SparseTensor)
    else:
        assert isinstance(output["feature2"], tf.Tensor)

    # tf.convert_to_tensor(
    test_case.assertAllClose(output["feature1"], feature1_dense)
    if only_selected_in_schema:
        test_case.assertAllClose(tf.sparse.to_dense(output["feature2"]), feature2_dense)
    else:
        test_case.assertAllClose(output["feature2"], feature2_dense)


@pytest.mark.parametrize(
    "only_selected_in_schema",
    [False, True],
)
def test_to_sparse(only_selected_in_schema):
    if only_selected_in_schema:
        schema = Schema(
            [
                create_categorical_column("feature1", tags=[Tags.CATEGORICAL], num_items=5),
                create_categorical_column("feature2", tags=[Tags.CATEGORICAL], num_items=5),
            ]
        )
        to_sparse = mm.ToSparse(schema.select_by_name("feature1"))
    else:
        # Converts all features to dense
        to_sparse = mm.ToSparse()

    feature1_dense = np.array([[1, 2, 3, 0], [0, 3, 1, 0]])
    feature2_dense = np.array([[2, 3, 4, 0], [2, 0, 1, 0]])

    data = {
        "feature1": feature1_dense,
        "feature2": feature2_dense,
    }

    output = to_sparse(data)

    assert output["feature1"].shape == data["feature1"].shape
    assert output["feature2"].shape == data["feature2"].shape

    assert isinstance(output["feature1"], tf.SparseTensor)
    if only_selected_in_schema:
        assert isinstance(output["feature2"], tf.Tensor)
    else:
        assert isinstance(output["feature2"], tf.SparseTensor)

    # tf.convert_to_tensor(
    tf.debugging.assert_equal(tf.sparse.to_dense(output["feature1"]), feature1_dense)
    if only_selected_in_schema:
        tf.debugging.assert_equal(output["feature2"], feature2_dense)
    else:
        tf.debugging.assert_equal(tf.sparse.to_dense(output["feature2"]), feature2_dense)
