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

from merlin.models.data.synthetic import SyntheticData

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin.models.tf")
test_utils = pytest.importorskip("merlin.models.tf.utils.testing_utils")


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
    with pytest.raises(Exception) as excinfo:
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


def test_dcn_v2_train_eval(ecommerce_data: SyntheticData, num_epochs=5, run_eagerly=True):
    dcn_body = (
        ml.InputBlock(
            ecommerce_data.schema,
            embedding_options=ml.EmbeddingOptions(embedding_dim_default=128),
            aggregation="concat",
        )
        .connect(ml.CrossBlock(3, low_rank_dim=64))
        .connect(ml.MLPBlock([512, 256]))
    )
    model = dcn_body.connect(ml.BinaryClassificationTask("click"))
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(ecommerce_data.tf_dataloader(batch_size=50), epochs=num_epochs)
    metrics = model.evaluate(*ecommerce_data.tf_features_and_targets, return_dict=True)
    test_utils.assert_binary_classification_loss_metrics(
        losses, metrics, target_name="click", num_epochs=num_epochs
    )


def test_dcn_v2_serialization(ecommerce_data: SyntheticData, run_eagerly=True):
    dcn_body = (
        ml.InputBlock(
            ecommerce_data.schema,
            embedding_options=ml.EmbeddingOptions(embedding_dim_default=128),
            aggregation="concat",
        )
        .connect(ml.CrossBlock(3, low_rank_dim=64))
        .connect(ml.MLPBlock([512, 256]))
    )
    model = dcn_body.connect(ml.BinaryClassificationTask("click"))
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    model.fit(ecommerce_data.dataset, batch_size=50, epochs=1)

    copy_model = test_utils.assert_serialization(model)
    test_utils.assert_loss_and_metrics_are_valid(copy_model, ecommerce_data.tf_features_and_targets)
