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

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin_models.tf")


@pytest.mark.parametrize("cross_layers", [1, 2, 3])
def test_cross(cross_layers):
    NUM_SEQS = 100
    DIM = 200

    input = tf.random.uniform((NUM_SEQS, DIM))
    cross = ml.CrossBlock(cross_layers)
    output = cross(input)

    assert list(output.shape) == [NUM_SEQS, DIM]
    assert not tf.reduce_all(tf.equal(input, output))


def test_cross_low_rank():
    NUM_SEQS = 100
    DIM = 200

    input = tf.random.uniform((NUM_SEQS, DIM))
    cross = ml.CrossBlock(depth=2, low_rank_dim=16)
    output = cross(input)

    assert list(output.shape) == [NUM_SEQS, DIM]
    assert not tf.reduce_all(tf.equal(input, output))


def test_cross_input_tuple_x0_xl():
    NUM_SEQS = 100
    DIM = 200

    x0 = tf.random.uniform((NUM_SEQS, DIM))
    x1 = tf.random.uniform((NUM_SEQS, DIM - 1))
    cross = ml.CrossBlock(3)
    with pytest.raises(ValueError) as excinfo:
        cross((x0, x1))
    assert "shapes mismatch" in str(excinfo.value)


def test_cross_0_layers():
    with pytest.raises(ValueError) as excinfo:
        ml.CrossBlock(depth=0)
    assert "Number of cross layers (depth) should be positive but is" in str(excinfo.value)


def test_cross_with_inputs_to_be_concat(testing_data: SyntheticData):
    inputs = ml.InputBlock(
        testing_data.schema,
        embedding_options=ml.EmbeddingOptions(embedding_dim_default=128),
    )
    cross = ml.CrossBlock(depth=1, inputs=inputs)
    output = cross(testing_data.tf_tensor_dict)

    assert list(output.shape) == [100, 518]


def test_dcn_v2_stacked(testing_data: SyntheticData):

    dcn_body = (
        ml.InputBlock(
            testing_data.schema,
            embedding_options=ml.EmbeddingOptions(embedding_dim_default=128),
            aggregation="concat",
        )
        .connect(ml.CrossBlock(3))
        .connect(ml.MLPBlock([512, 256]))
    )

    output = dcn_body(testing_data.tf_tensor_dict)

    assert list(output.shape) == [100, 256]


def test_dcn_v2_stacked_low_rank(testing_data: SyntheticData):

    dcn_body = (
        ml.InputBlock(
            testing_data.schema,
            embedding_options=ml.EmbeddingOptions(embedding_dim_default=128),
            aggregation="concat",
        )
        .connect(ml.CrossBlock(3, low_rank_dim=64))
        .connect(ml.MLPBlock([512, 256]))
    )

    output = dcn_body(testing_data.tf_tensor_dict)

    assert list(output.shape) == [100, 256]


def test_dcn_v2_parallel(testing_data: SyntheticData):
    input_layer = ml.InputBlock(
        testing_data.schema,
        embedding_options=ml.EmbeddingOptions(embedding_dim_default=128),
        aggregation="concat",
    )

    concat_input_dim = input_layer(testing_data.tf_tensor_dict).shape[-1]
    mlp_layers = [512, 256]
    dcn_body = input_layer.connect_branch(
        ml.CrossBlock(3), ml.MLPBlock(mlp_layers), aggregation="concat"
    )

    output = dcn_body(testing_data.tf_tensor_dict)

    assert list(output.shape) == [100, concat_input_dim + mlp_layers[-1]]
