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

from merlin_models.data.synthetic import SyntheticData
from merlin_standard_lib import Tag

tr = pytest.importorskip("merlin_models.tf")


def test_dlrm_block_yoochoose(testing_data: SyntheticData):

    dlrm = tr.DLRMBlock(
        testing_data.schema, bottom_block=tr.MLPBlock([64]), top_block=tr.MLPBlock([32])
    )
    outputs = dlrm(testing_data.tf_tensor_dict)

    assert list(outputs.shape) == [100, 32]


def test_dlrm_block_no_continuous_features(testing_data):
    schema = testing_data.schema.remove_by_tag(Tag.CONTINUOUS)
    dlrm = tr.DLRMBlock(schema, bottom_block=tr.MLPBlock([64]), top_block=tr.MLPBlock([32]))
    outputs = dlrm(testing_data.tf_tensor_dict)

    assert list(outputs.shape) == [100, 32]


def test_dlrm_block_no_categ_features(testing_data):
    schema = testing_data.schema.remove_by_tag(Tag.CATEGORICAL)
    dlrm = tr.DLRMBlock(schema, bottom_block=tr.MLPBlock([64]), top_block=tr.MLPBlock([32]))
    outputs = dlrm(testing_data.tf_tensor_dict)

    assert list(outputs.shape) == [100, 32]


def test_dlrm_block_single_categ_feature(testing_data):
    schema = testing_data.schema.select_by_tag(Tag.ITEM_ID)
    dlrm = tr.DLRMBlock(schema, bottom_block=tr.MLPBlock([64]), top_block=tr.MLPBlock([32]))
    outputs = dlrm(testing_data.tf_tensor_dict)

    assert list(outputs.shape) == [100, 32]


def test_dlrm_block_single_continuous_feature(testing_data):
    schema = testing_data.schema.select_by_name("event_hour_sin")
    dlrm = tr.DLRMBlock(schema, bottom_block=tr.MLPBlock([64]), top_block=tr.MLPBlock([16]))
    outputs = dlrm(testing_data.tf_tensor_dict)

    assert list(outputs.shape) == [100, 16]


def test_dlrm_block_no_schema():
    with pytest.raises(ValueError) as excinfo:
        tr.DLRMBlock(schema=None, bottom_block=tr.MLPBlock([64]), top_block=tr.MLPBlock([32]))
    assert "The schema is required by DLRM" in str(excinfo.value)


def test_dlrm_block_no_bottom_block(tabular_schema):
    with pytest.raises(ValueError) as excinfo:
        tr.DLRMBlock(schema=tabular_schema, bottom_block=None)
    assert "The bottom_block is required by DLRM" in str(excinfo.value)


def test_dlrm_emb_dim_do_not_match_bottom_mlp(tabular_schema):
    with pytest.raises(ValueError) as excinfo:
        tr.DLRMBlock(schema=tabular_schema, bottom_block=tr.MLPBlock([64]), embedding_dim=75)
    assert "needs to match the last layer of bottom MLP" in str(excinfo.value)
