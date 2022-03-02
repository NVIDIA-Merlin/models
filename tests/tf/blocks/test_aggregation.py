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

import merlin.models.tf as ml
from merlin.models.data.synthetic import SyntheticData
from merlin.models.tf.blocks.aggregation import ElementWiseMultiply
from merlin.schema import Tags


def test_concat_aggregation_yoochoose(testing_data: SyntheticData):
    tab_module = ml.InputBlock(testing_data.schema)

    block = tab_module >> ml.ConcatFeatures()

    out = block(testing_data.tf_tensor_dict)

    assert out.shape[-1] == 262


def test_stack_aggregation_yoochoose(testing_data: SyntheticData):
    tab_module = ml.EmbeddingFeatures.from_schema(testing_data.schema)

    block = tab_module >> ml.StackFeatures()

    out = block(testing_data.tf_tensor_dict)

    assert out.shape[1] == 64
    assert out.shape[2] == 4


def test_element_wise_sum_features_different_shapes():
    with pytest.raises(ValueError) as excinfo:
        element_wise_op = ml.ElementwiseSum()
        input = {
            "item_id/list": tf.random.uniform((10, 20)),
            "category/list": tf.random.uniform((10, 25)),
        }
        element_wise_op(input)
    assert "shapes of all input features are not equal" in str(excinfo.value)


def test_element_wise_sum_aggregation_yoochoose(testing_data: SyntheticData):
    tab_module = ml.EmbeddingFeatures.from_schema(testing_data.schema)

    block = tab_module >> ml.ElementwiseSum()

    out = block(testing_data.tf_tensor_dict)

    assert out.shape[-1] == 64


def test_element_wise_sum_item_multi_no_col_group():
    with pytest.raises(ValueError) as excinfo:
        element_wise_op = ml.ElementwiseSumItemMulti()
        element_wise_op(None)
    assert "requires a schema." in str(excinfo.value)


def test_element_wise_sum_item_multi_col_group_no_item_id(testing_data: SyntheticData):
    with pytest.raises(ValueError) as excinfo:
        categ_schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)
        # Remove the item id from col_group
        categ_schema = categ_schema.without("item_id")
        element_wise_op = ml.ElementwiseSumItemMulti(categ_schema)
        element_wise_op(None)
    assert "no column" in str(excinfo.value)


def test_element_wise_sum_item_multi_features_different_shapes(testing_data: SyntheticData):
    with pytest.raises(ValueError) as excinfo:
        categ_schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)
        element_wise_op = ml.ElementwiseSumItemMulti(categ_schema)
        input = {
            "item_id": tf.random.uniform((10, 20)),
            "category": tf.random.uniform((10, 25)),
        }
        element_wise_op(input)
    assert "shapes of all input features are not equal" in str(excinfo.value)


def test_element_wise_sum_item_multi_aggregation_yoochoose(testing_data: SyntheticData):
    tab_module = ml.EmbeddingFeatures.from_schema(testing_data.schema)

    block = tab_module >> ml.ElementwiseSumItemMulti(testing_data.schema)

    out = block(testing_data.tf_tensor_dict)

    assert out.shape[-1] == 64


def test_elementwisemultiply():
    emb1 = np.random.uniform(-1, 1, size=(5, 10))
    emb2 = np.random.uniform(-1, 1, size=(5, 10))
    x = ElementWiseMultiply()(tf.constant(emb1), tf.constant(emb2))

    assert np.mean(np.isclose(x.numpy(), np.multiply(emb1, emb2))) == 1
    assert x.numpy().shape == (5, 10)


# def test_element_wise_sum_item_multi_aggregation_registry_yoochoose(
#     yoochoose_schema, tf_yoochoose_like
# ):
#     tab_module = tr.TabularSequenceFeatures.from_schema(
#         yoochoose_schema.select_by_tag(Tags.CATEGORICAL),
#          aggregation="element-wise-sum-item-multi"
#     )
#
#     out = tab_module(tf_yoochoose_like)
#
#     assert out.shape[-1] == 64
