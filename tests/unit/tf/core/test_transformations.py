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

import tempfile

import pytest
import tensorflow as tf
from tensorflow.test import TestCase

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.models.tf.core.combinators import ParallelBlock, TabularBlock
from merlin.models.tf.utils import testing_utils
from merlin.models.utils.schema_utils import create_categorical_column, create_continuous_column
from merlin.schema import Schema, Tags


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

    expand_dims_op = ml.ExpandDims(expand_dims=-1)
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

    expand_dims_op = ml.ExpandDims(expand_dims={"cont_feat2": 0, "multi_hot_categ_feat": 1})
    expanded_inputs = expand_dims_op(inputs)

    assert inputs.keys() == expanded_inputs.keys()

    assert list(expanded_inputs["cont_feat1"].shape) == [NUM_ROWS]
    assert list(expanded_inputs["cont_feat2"].shape) == [1, NUM_ROWS]
    assert list(expanded_inputs["multi_hot_categ_feat"].shape) == [NUM_ROWS, 1, 4]


def test_categorical_one_hot_encoding():
    NUM_ROWS = 100
    MAX_LEN = 4

    s = Schema(
        [
            create_categorical_column("cat1", num_items=200, tags=[Tags.CATEGORICAL]),
            create_categorical_column("cat2", num_items=1000, tags=[Tags.CATEGORICAL]),
            create_categorical_column("cat3", num_items=50, tags=[Tags.CATEGORICAL]),
            create_continuous_column("cont1", min_value=0, max_value=1, tags=[Tags.CONTINUOUS]),
        ]
    )

    cardinalities = {"cat1": 201, "cat2": 1001, "cat3": 51}
    inputs = {}
    for cat, cardinality in cardinalities.items():
        inputs[cat] = tf.random.uniform((NUM_ROWS, 1), minval=1, maxval=cardinality, dtype=tf.int32)
    inputs["cat3"] = tf.random.uniform(
        (NUM_ROWS, MAX_LEN), minval=1, maxval=cardinalities["cat3"], dtype=tf.int32
    )
    inputs["cont1"] = tf.random.uniform((NUM_ROWS, 1), minval=0, maxval=1, dtype=tf.float32)

    input_shape = {}
    for key in inputs:
        input_shape[key] = inputs[key].shape

    categorical_one_hot = ml.CategoricalOneHot(schema=s)
    outputs = categorical_one_hot(inputs)
    outputs_shape = categorical_one_hot.compute_output_shape(input_shape)

    assert list(outputs["cat1"].shape) == [NUM_ROWS, 201]
    assert list(outputs["cat2"].shape) == [NUM_ROWS, 1001]
    assert list(outputs["cat3"].shape) == [NUM_ROWS, MAX_LEN, 51]

    assert inputs["cat1"][0].numpy() == tf.where(outputs["cat1"][0, :] == 1).numpy()[0]
    assert list(outputs.keys()) == ["cat1", "cat2", "cat3"]

    assert outputs_shape["cat1"] == outputs["cat1"].shape
    assert outputs_shape["cat2"] == outputs["cat2"].shape
    assert outputs_shape["cat3"] == outputs["cat3"].shape


@pytest.mark.parametrize(
    "input",
    [
        tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]),
        tf.ragged.constant([[9, 8, 7], [], [6, 5], [4]]),
    ],
)
def test_categorical_one_hot_invalid_input(input):
    test_case = TestCase()
    s = Schema(
        [
            create_categorical_column("cat1", num_items=10, tags=[Tags.CATEGORICAL]),
        ]
    )
    inputs = {}
    inputs["cat1"] = input
    categorical_one_hot = ml.CategoricalOneHot(schema=s)
    with test_case.assertRaisesRegex(ValueError, "inputs should be a Tensor"):
        categorical_one_hot(inputs)


def test_categorical_one_hot_from_config():
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=20),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=20),
            create_continuous_column("cont1", min_value=0, max_value=1, tags=[Tags.CONTINUOUS]),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant([[1], [2], [3], [2], [1]])
    inputs["cat2"] = tf.constant([101, 101, 103, 102, 102])
    inputs["cont1"] = tf.random.uniform((5, 1), minval=0, maxval=1, dtype=tf.float32)

    input_shape = {}
    for key in inputs:
        input_shape[key] = inputs[key].shape

    categorical_one_hot = ml.CategoricalOneHot(schema=schema)
    outputs = categorical_one_hot(inputs)
    output_shape = categorical_one_hot.compute_output_shape(input_shape)

    cloned_categorical_one_hot = ml.CategoricalOneHot.from_config(categorical_one_hot.get_config())
    cloned_outputs = cloned_categorical_one_hot(inputs)
    cloned_output_shape = cloned_categorical_one_hot.compute_output_shape(input_shape)

    test_case.assertAllClose(cloned_output_shape, output_shape)
    test_case.assertAllClose(cloned_outputs, outputs)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_categorical_one_hot_as_pre(ecommerce_data: Dataset, run_eagerly):
    schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    body = ParallelBlock(
        TabularBlock.from_schema(schema=schema, pre=ml.CategoricalOneHot(schema)),
        is_input=True,
    ).connect(ml.MLPBlock([32]))
    model = ml.Model(body, ml.BinaryClassificationTask("click"))

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    testing_utils.model_test(model, ecommerce_data)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_categorical_one_hot_in_model(ecommerce_data: Dataset, run_eagerly):
    schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    branches = {
        "one_hot": ml.CategoricalOneHot(schema, is_input=True),
        "features": ml.InputBlock(ecommerce_data.schema),
    }
    body = ParallelBlock(branches, is_input=True).connect(ml.MLPBlock([32]))
    model = ml.Model(body, ml.BinaryClassificationTask("click"))

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    testing_utils.model_test(model, ecommerce_data)


def test_popularity_logits_correct():
    from merlin.models.tf.core.base import PredictionOutput
    from merlin.models.tf.core.transformations import PopularityLogitsCorrection

    schema = Schema(
        [
            create_categorical_column(
                "item_feature", num_items=100, tags=[Tags.CATEGORICAL, Tags.ITEM_ID]
            ),
        ]
    )

    NUM_ITEMS = 101
    NUM_ROWS = 16
    NUM_SAMPLE = 20

    logits = tf.random.uniform((NUM_ROWS, NUM_SAMPLE))
    negative_item_ids = tf.random.uniform(
        (NUM_SAMPLE - 1,), minval=1, maxval=NUM_ITEMS, dtype=tf.int32
    )
    positive_item_ids = tf.random.uniform((NUM_ROWS,), minval=1, maxval=NUM_ITEMS, dtype=tf.int32)
    item_frequency = tf.sort(tf.random.uniform((NUM_ITEMS,), minval=0, maxval=1000, dtype=tf.int32))

    inputs = PredictionOutput(
        predictions=logits,
        targets=[],
        positive_item_ids=positive_item_ids,
        negative_item_ids=negative_item_ids,
    )

    corrected_logits = PopularityLogitsCorrection(
        item_frequency, reg_factor=0.5, schema=schema
    ).call_outputs(outputs=inputs)

    tf.debugging.assert_less_equal(logits, corrected_logits.predictions)


def test_popularity_logits_correct_from_parquet():
    import numpy as np
    import pandas as pd

    from merlin.models.tf.core.transformations import PopularityLogitsCorrection

    schema = Schema(
        [
            create_categorical_column(
                "item_feature", num_items=100, tags=[Tags.CATEGORICAL, Tags.ITEM_ID]
            ),
        ]
    )
    NUM_ITEMS = 101

    frequency_table = pd.DataFrame(
        {"frequency": list(np.sort(np.random.randint(0, 1000, size=(NUM_ITEMS,))))}
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        frequency_table.to_parquet(tmpdir + "/frequency_table.parquet")
        corrected_logits = PopularityLogitsCorrection.from_parquet(
            tmpdir + "/frequency_table.parquet",
            frequencies_probs_col="frequency",
            gpu=False,
            schema=schema,
        )
    assert corrected_logits.get_candidate_probs().shape == (NUM_ITEMS,)


def test_items_weight_tying_with_different_domain_name():
    from merlin.models.tf.core.transformations import ItemsPredictionWeightTying

    NUM_ROWS = 16
    schema = Schema(
        [
            create_categorical_column(
                "item_id",
                domain_name="joint_item_id",
                num_items=100,
                tags=[Tags.CATEGORICAL, Tags.ITEM_ID],
            ),
        ]
    )
    inputs = {
        "item_id": tf.random.uniform((NUM_ROWS, 1), minval=1, maxval=101, dtype=tf.int32),
        "target": tf.random.uniform((NUM_ROWS, 1), minval=0, maxval=10, dtype=tf.int32),
    }

    weight_tying_block = ItemsPredictionWeightTying(schema=schema)
    input_block = ml.InputBlock(schema)
    task = ml.MultiClassClassificationTask("target")
    model = ml.Model(input_block, ml.MLPBlock([64]), weight_tying_block, task)

    _ = model(inputs)
    weight_tying_embeddings = model.blocks[2].context.get_embedding("joint_item_id")
    assert weight_tying_embeddings.shape == (101, 64)


def test_hashedcross_scalars():
    test_case = TestCase()
    schema = Schema(
        [
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=3),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=3),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant("A")
    inputs["cat2"] = tf.constant(101)
    hashed_cross_op = ml.HashedCross(schema=schema, num_bins=10)
    outputs = hashed_cross_op(inputs)
    output_name, output_value = outputs.popitem()

    assert output_name == "cross_cat1_cat2"
    assert output_value.shape.as_list() == []
    test_case.assertAllClose(output_value, 1)


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
    hashed_cross_op = ml.HashedCross(schema=schema, num_bins=10)
    outputs = hashed_cross_op(inputs)
    _, output_value = outputs.popitem()

    assert output_value.shape.as_list() == [5]
    test_case.assertAllClose(output_value, [1, 4, 1, 6, 3])


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
    hashed_cross_op = ml.HashedCross(schema=schema, num_bins=10)
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
    hashed_cross = ml.HashedCross(schema=schema, num_bins=10)
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
    hashed_cross = ml.HashedCross(
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
    hashed_cross_op = ml.HashedCross(schema=schema, num_bins=4, output_mode="one_hot", sparse=True)
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

    hashed_cross_op = ml.HashedCross(schema=schema, num_bins=4, output_mode="one_hot", sparse=True)
    outputs = hashed_cross_op(inputs)
    _, output_value = outputs.popitem()
    assert isinstance(output_value, tf.SparseTensor) is True
    assert output_value.shape.as_list() == [7, 4]

    hashed_cross_op = ml.HashedCross(schema=schema, num_bins=4, output_mode="one_hot", sparse=False)
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
    hashed_cross_op = ml.HashedCross(schema=schema, num_bins=5, output_mode="one_hot", sparse=True)
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
        ml.HashedCross(num_bins=10, schema=schema)([tf.constant(1)])


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
    hashed_cross_op = ml.HashedCross(schema=schema, num_bins=5, output_mode="one_hot", sparse=False)
    cloned_hashed_cross_op = ml.HashedCross.from_config(hashed_cross_op.get_config())
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
    hashed_cross_0 = ml.HashedCross(
        schema=schema_0, num_bins=5, output_mode="one_hot", sparse=True, output_name="cross_0"
    )
    hashed_cross_1 = ml.HashedCross(
        schema=schema_1, num_bins=10, output_mode="one_hot", sparse=True, output_name="cross_1"
    )
    hashed_crosses = ParallelBlock([hashed_cross_0, hashed_cross_1])
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
    body = ParallelBlock(
        TabularBlock.from_schema(
            schema=cross_schema, pre=ml.HashedCross(cross_schema, num_bins=1000)
        ),
        is_input=True,
    ).connect(ml.MLPBlock([64]))
    model = ml.Model(body, ml.BinaryClassificationTask("click"))

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    testing_utils.model_test(model, ecommerce_data)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_hashedcross_in_model(ecommerce_data: Dataset, run_eagerly):
    cross_schema = ecommerce_data.schema.select_by_name(names=["user_categories", "item_category"])
    branches = {
        "cross_product": ml.HashedCross(cross_schema, num_bins=1000, is_input=True),
        "features": ml.InputBlock(ecommerce_data.schema),
    }
    body = ParallelBlock(branches, is_input=True).connect(ml.MLPBlock([64]))
    model = ml.Model(body, ml.BinaryClassificationTask("click"))

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    testing_utils.model_test(model, ecommerce_data)


def test_hashedcrossall():
    schema = Schema(
        [
            # num_items: 0, 1, 2 thus cardinality = 3
            create_categorical_column("cat1", tags=[Tags.CATEGORICAL], num_items=2),
            create_categorical_column("cat2", tags=[Tags.CATEGORICAL], num_items=2),
            create_categorical_column("cat3", tags=[Tags.CATEGORICAL], num_items=2),
        ]
    )
    inputs = {}
    inputs["cat1"] = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
    inputs["cat2"] = tf.constant([[101], [101], [101], [102], [102]])
    inputs["cat3"] = tf.constant([[1], [0], [1], [2], [2]])

    hashed_cross_all = ml.HashedCrossAll(
        schema=schema,
        infer_num_bins=True,
        output_mode="one_hot",
        sparse=True,
        max_num_bins=25,
        max_level=3,
    )

    outputs = hashed_cross_all(inputs)
    assert len(outputs) == 4

    output_value_0 = outputs["cross_cat1_cat2"]
    assert output_value_0.shape.as_list() == [5, 9]

    output_value_1 = outputs["cross_cat1_cat2_cat3"]
    assert output_value_1.shape.as_list() == [5, 25]


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_hashedcrossall_in_model(ecommerce_data: Dataset, run_eagerly):
    cross_schema = ecommerce_data.schema.select_by_name(
        names=["user_categories", "item_category", "item_brand"]
    )
    branches = {
        "cross_product": ml.HashedCrossAll(cross_schema, max_num_bins=1000, infer_num_bins=True),
        "features": ml.InputBlock(ecommerce_data.schema),
    }
    body = ParallelBlock(branches, is_input=True).connect(ml.MLPBlock([64]))
    model = ml.Model(body, ml.BinaryClassificationTask("click"))

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    testing_utils.model_test(model, ecommerce_data)
