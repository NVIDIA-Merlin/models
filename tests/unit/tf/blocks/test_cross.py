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
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("cross_layers", [1, 2, 3])
def test_cross(cross_layers):
    NUM_SEQS = 100
    DIM = 200

    input = tf.random.uniform((NUM_SEQS, DIM))
    cross = mm.CrossBlock(cross_layers)
    output = cross(input)

    assert list(output.shape) == [NUM_SEQS, DIM]
    assert not tf.reduce_all(tf.equal(input, output))


def test_cross_low_rank():
    NUM_SEQS = 100
    DIM = 200

    input = tf.random.uniform((NUM_SEQS, DIM))
    cross = mm.CrossBlock(depth=2, low_rank_dim=16)
    output = cross(input)

    assert list(output.shape) == [NUM_SEQS, DIM]
    assert not tf.reduce_all(tf.equal(input, output))


def test_cross_input_tuple_x0_xl():
    NUM_SEQS = 100
    DIM = 200

    x0 = tf.random.uniform((NUM_SEQS, DIM))
    x1 = tf.random.uniform((NUM_SEQS, DIM - 1))
    cross = mm.CrossBlock(3)
    with pytest.raises(Exception) as excinfo:
        cross((x0, x1))
    assert "shapes mismatch" in str(excinfo.value)


def test_cross_0_layers():
    with pytest.raises(ValueError) as excinfo:
        mm.CrossBlock(depth=0)
    assert "Number of cross layers (depth) should be positive but is" in str(excinfo.value)


def test_cross_with_inputs_to_be_concat(testing_data: Dataset):
    inputs = mm.InputBlock(
        testing_data.schema,
        embedding_options=mm.EmbeddingOptions(embedding_dim_default=128),
    )
    cross = mm.CrossBlock(depth=1, inputs=inputs)
    output = cross(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(output.shape) == [100, 518]


def test_dcn_v2_stacked(testing_data: Dataset):

    dcn_body = (
        mm.InputBlock(
            testing_data.schema,
            embedding_options=mm.EmbeddingOptions(embedding_dim_default=128),
            aggregation="concat",
        )
        .connect(mm.CrossBlock(3))
        .connect(mm.MLPBlock([512, 256]))
    )

    output = dcn_body(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(output.shape) == [100, 256]


def test_dcn_v2_stacked_low_rank(testing_data: Dataset):

    dcn_body = (
        mm.InputBlock(
            testing_data.schema,
            embedding_options=mm.EmbeddingOptions(embedding_dim_default=128),
            aggregation="concat",
        )
        .connect(mm.CrossBlock(3, low_rank_dim=64))
        .connect(mm.MLPBlock([512, 256]))
    )

    output = dcn_body(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(output.shape) == [100, 256]


def test_dcn_v2_parallel(testing_data: Dataset):
    input_layer = mm.InputBlock(
        testing_data.schema,
        embedding_options=mm.EmbeddingOptions(embedding_dim_default=128),
        aggregation="concat",
    )

    features = mm.sample_batch(testing_data, batch_size=100, include_targets=False)
    concat_input_dim = input_layer(features).shape[-1]
    mlp_layers = [512, 256]
    dcn_body = input_layer.connect_branch(
        mm.CrossBlock(3), mm.MLPBlock(mlp_layers), aggregation="concat"
    )

    output = dcn_body(features)

    assert list(output.shape) == [100, concat_input_dim + mlp_layers[-1]]


def test_dcn_v2(ecommerce_data: Dataset, run_eagerly=True):
    dcn_body = (
        mm.InputBlock(
            ecommerce_data.schema,
            embedding_options=mm.EmbeddingOptions(embedding_dim_default=128),
            aggregation="concat",
        )
        .connect(mm.CrossBlock(3, low_rank_dim=64))
        .connect(mm.MLPBlock([512, 256]))
    )
    model = mm.Model(dcn_body, mm.BinaryClassificationTask("click"))
    testing_utils.model_test(model, ecommerce_data)
