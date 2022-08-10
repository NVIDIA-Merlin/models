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

import pytest
import torch

import merlin.models.torch as ml
from merlin.schema import Tags


def test_concat_aggregation_yoochoose(tabular_schema, torch_tabular_data):
    schema = tabular_schema
    tab_module = ml.features.tabular.TabularFeatures.from_schema(schema)

    block = tab_module >> ml.ConcatFeatures()

    out = block(torch_tabular_data)

    assert out.shape[-1] == 262


def test_stack_aggregation_yoochoose(tabular_schema, torch_tabular_data):
    schema = tabular_schema
    tab_module = ml.EmbeddingFeatures.from_schema(schema)

    block = tab_module >> ml.StackFeatures()

    out = block(torch_tabular_data)

    assert out.shape[1] == 64
    assert out.shape[2] == 4


def test_element_wise_sum_features_different_shapes():
    with pytest.raises(ValueError) as excinfo:
        element_wise_op = ml.ElementwiseSum()
        input = {
            "item_id/list": torch.rand(10, 20),
            "category/list": torch.rand(10, 25),
        }
        element_wise_op(input)
    assert "shapes of all input features are not equal" in str(excinfo.value)


def test_element_wise_sum_aggregation_yoochoose(tabular_schema, torch_tabular_data):
    schema = tabular_schema
    tab_module = ml.EmbeddingFeatures.from_schema(schema)

    block = tab_module >> ml.ElementwiseSum()

    out = block(torch_tabular_data)

    assert out.shape[-1] == 64


def test_element_wise_sum_item_multi_no_col_group():
    with pytest.raises(ValueError) as excinfo:
        element_wise_op = ml.ElementwiseSumItemMulti()
        element_wise_op(None)
    assert "requires a schema" in str(excinfo.value)


def test_element_wise_sum_item_multi_col_group_no_item_id(tabular_schema):
    with pytest.raises(ValueError) as excinfo:
        categ_schema = tabular_schema.select_by_tag(Tags.CATEGORICAL)
        # Remove the item id from col_group
        categ_schema = categ_schema.without("item_id")
        element_wise_op = ml.ElementwiseSumItemMulti(categ_schema)
        element_wise_op(None)
    assert "no column" in str(excinfo.value)


def test_element_wise_sum_item_multi_features_different_shapes(tabular_schema):
    with pytest.raises(ValueError) as excinfo:
        categ_schema = tabular_schema.select_by_tag(Tags.CATEGORICAL)
        element_wise_op = ml.ElementwiseSumItemMulti(categ_schema)
        input = {
            "item_id": torch.rand(10, 20),
            "categories": torch.rand(10, 25),
        }
        element_wise_op(input)
    assert "shapes of all input features are not equal" in str(excinfo.value)


def test_element_wise_sum_item_multi_aggregation_yoochoose(tabular_schema, torch_tabular_data):
    schema = tabular_schema
    tab_module = ml.EmbeddingFeatures.from_schema(schema)

    block = tab_module >> ml.ElementwiseSumItemMulti(schema)

    out = block(torch_tabular_data)

    assert out.shape[-1] == 64
