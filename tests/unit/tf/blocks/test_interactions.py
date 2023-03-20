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
from merlin.io import Dataset
from merlin.schema import Tags


def test_fm_pairwise_interaction():
    NUM_ROWS = 100
    NUM_FEATS = 10
    EMBED_DIM = 64

    inputs = tf.random.uniform((NUM_ROWS, NUM_FEATS, EMBED_DIM))

    pairwise_interaction = mm.FMPairwiseInteraction()
    outputs = pairwise_interaction(inputs)

    assert list(outputs.shape) == [NUM_ROWS, EMBED_DIM]


def test_fm_block(ecommerce_data: Dataset):
    schema = ecommerce_data.schema.remove_by_tag(Tags.TARGET)

    fm_block = mm.FMBlock(
        schema,
        factors_dim=32,
    )

    batch = mm.sample_batch(ecommerce_data, batch_size=16, include_targets=False)
    output = fm_block(batch)
    output.shape.as_list() == [16, 1]


def test_fm_block_with_multi_hot_categ_features(testing_data: Dataset):
    schema = testing_data.schema
    cat_schema = schema.select_by_tag(Tags.CATEGORICAL)
    cat_schema_onehot = cat_schema.remove_by_tag(Tags.LIST)
    cat_schema_multihot = cat_schema.select_by_tag(Tags.LIST)

    input_block = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            cat_schema,
            dim=32,
            sequence_combiner="mean",
        ),
        aggregation=None,
    )

    wide_input_block = mm.ParallelBlock(
        {
            "categorical_ohe": mm.Filter(cat_schema_onehot).connect(
                mm.CategoryEncoding(cat_schema_onehot, sparse=True, output_mode="one_hot"),
            ),
            "categorical_mhe": mm.SequentialBlock(
                mm.Filter(cat_schema_multihot),
                mm.ToDense(cat_schema_multihot),
                mm.CategoryEncoding(cat_schema_multihot, sparse=True, output_mode="multi_hot"),
            ),
            "continuous": mm.SequentialBlock(
                mm.Filter(schema.select_by_tag(Tags.CONTINUOUS)), mm.ToSparse()
            ),
        },
        aggregation="concat",
    )

    fm_block = mm.FMBlock(
        schema,
        fm_input_block=input_block,
        wide_input_block=wide_input_block,
        factors_dim=32,
    )

    batch, _ = mm.sample_batch(testing_data, batch_size=16, prepare_features=True)

    output = fm_block(batch)
    assert output.shape.as_list() == [16, 1]
