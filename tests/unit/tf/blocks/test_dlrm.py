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

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.schema import Tags


def test_dlrm_block(testing_data: Dataset):
    schema = testing_data.schema
    dlrm = mm.DLRMBlock(
        schema,
        embedding_dim=64,
        bottom_block=mm.MLPBlock([64]),
        top_block=mm.DenseResidualBlock(),
    )
    features = mm.sample_batch(testing_data, batch_size=100, include_targets=False)
    outputs = dlrm(features)
    num_features = len(schema.select_by_tag(Tags.CATEGORICAL)) + 1
    dot_product_dim = (num_features - 1) * num_features // 2
    assert list(outputs.shape) == [100, dot_product_dim + 64]


def test_dlrm_block_no_top_block(testing_data: Dataset):
    schema = testing_data.schema
    dlrm = mm.DLRMBlock(
        schema,
        embedding_dim=64,
        bottom_block=mm.MLPBlock([64]),
    )
    outputs = dlrm(mm.sample_batch(testing_data, batch_size=100, include_targets=False))
    num_features = len(schema.select_by_tag(Tags.CATEGORICAL)) + 1
    dot_product_dim = (num_features - 1) * num_features // 2

    assert list(outputs.shape) == [100, dot_product_dim]


def test_dlrm_block_no_continuous_features(testing_data: Dataset):
    schema = testing_data.schema.remove_by_tag(Tags.CONTINUOUS)
    dlrm = mm.DLRMBlock(schema, embedding_dim=64, top_block=mm.MLPBlock([32]))
    outputs = dlrm(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 32]


def test_dlrm_block_no_categ_features(testing_data: Dataset):
    schema = testing_data.schema.remove_by_tag(Tags.CATEGORICAL)
    with pytest.raises(ValueError) as excinfo:
        mm.DLRMBlock(
            schema, embedding_dim=64, bottom_block=mm.MLPBlock([64]), top_block=mm.MLPBlock([16])
        )
    assert "DLRM requires categorical features" in str(excinfo.value)


def test_dlrm_block_single_categ_feature(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag([Tags.ITEM_ID])
    dlrm = mm.DLRMBlock(schema, embedding_dim=64, top_block=mm.MLPBlock([32]))
    outputs = dlrm(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 32]


def test_dlrm_block_no_schema():
    with pytest.raises(ValueError) as excinfo:
        mm.DLRMBlock(
            schema=None,
            embedding_dim=64,
            bottom_block=mm.MLPBlock([64]),
            top_block=mm.MLPBlock([32]),
        )
    assert "The schema is required by DLRM" in str(excinfo.value)


def test_dlrm_block_no_bottom_block(testing_data: Dataset):
    with pytest.raises(ValueError) as excinfo:
        mm.DLRMBlock(schema=testing_data.schema, embedding_dim=64, bottom_block=None)
    assert "The bottom_block is required by DLRM" in str(excinfo.value)


def test_dlrm_emb_dim_do_not_match_bottom_mlp(testing_data: Dataset):
    with pytest.raises(ValueError) as excinfo:
        mm.DLRMBlock(schema=testing_data.schema, bottom_block=mm.MLPBlock([64]), embedding_dim=75)
    assert "needs to match the last layer of bottom MLP" in str(excinfo.value)


def test_dlrm_raises_with_embeddings_and_options(testing_data: Dataset):
    schema = testing_data.schema
    with pytest.raises(ValueError) as excinfo:
        mm.DLRMBlock(
            schema,
            embedding_dim=10,
            embedding_options=mm.EmbeddingOptions(),
            embeddings=mm.Embeddings(schema.select_by_tag(Tags.CATEGORICAL)),
        )
    assert "Only one-of `embeddings` or `embedding_options` may be provided" in str(excinfo.value)


def test_dlrm_with_embeddings(testing_data: Dataset):
    schema = testing_data.schema
    embedding_dim = 12
    top_dim = 4
    dlrm = mm.DLRMBlock(
        schema,
        embeddings=mm.Embeddings(schema.select_by_tag(Tags.CATEGORICAL), dim=embedding_dim),
        bottom_block=mm.MLPBlock([embedding_dim]),
        top_block=mm.MLPBlock([top_dim]),
    )
    outputs = dlrm(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 4]
