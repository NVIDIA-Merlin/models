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

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.schema import Tags


def test_dlrm_block(testing_data: Dataset):
    dlrm = ml.DLRMBlock(
        testing_data.schema,
        embedding_dim=64,
        bottom_block=ml.MLPBlock([64]),
        top_block=ml.DenseResidualBlock(),
    )
    outputs = dlrm(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 2080]


def test_dlrm_block_no_top_block(testing_data: Dataset):
    dlrm = ml.DLRMBlock(
        testing_data.schema,
        embedding_dim=64,
        bottom_block=ml.MLPBlock([64]),
    )
    outputs = dlrm(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 2016]


def test_dlrm_block_no_continuous_features(testing_data: Dataset):
    schema = testing_data.schema.remove_by_tag(Tags.CONTINUOUS)
    dlrm = ml.DLRMBlock(schema, embedding_dim=64, top_block=ml.MLPBlock([32]))
    outputs = dlrm(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 32]


def test_dlrm_block_no_categ_features(testing_data: Dataset):
    schema = testing_data.schema.remove_by_tag(Tags.CATEGORICAL)
    with pytest.raises(ValueError) as excinfo:
        ml.DLRMBlock(
            schema, embedding_dim=64, bottom_block=ml.MLPBlock([64]), top_block=ml.MLPBlock([16])
        )
    assert "DLRM requires categorical features" in str(excinfo.value)


def test_dlrm_block_single_categ_feature(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag([Tags.ITEM_ID])
    dlrm = ml.DLRMBlock(schema, embedding_dim=64, top_block=ml.MLPBlock([32]))
    outputs = dlrm(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 32]


def test_dlrm_block_no_schema():
    with pytest.raises(ValueError) as excinfo:
        ml.DLRMBlock(
            schema=None,
            embedding_dim=64,
            bottom_block=ml.MLPBlock([64]),
            top_block=ml.MLPBlock([32]),
        )
    assert "The schema is required by DLRM" in str(excinfo.value)


def test_dlrm_block_no_bottom_block(testing_data: Dataset):
    with pytest.raises(ValueError) as excinfo:
        ml.DLRMBlock(schema=testing_data.schema, embedding_dim=64, bottom_block=None)
    assert "The bottom_block is required by DLRM" in str(excinfo.value)


def test_dlrm_emb_dim_do_not_match_bottom_mlp(testing_data: Dataset):
    with pytest.raises(ValueError) as excinfo:
        ml.DLRMBlock(schema=testing_data.schema, bottom_block=ml.MLPBlock([64]), embedding_dim=75)
    assert "needs to match the last layer of bottom MLP" in str(excinfo.value)
