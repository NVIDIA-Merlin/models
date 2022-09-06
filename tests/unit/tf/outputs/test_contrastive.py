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
from merlin.models.tf.core.transformations import RenameFeatures
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.outputs.sampling.popularity import PopularityBasedSamplerV2
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_contrastive_mf(ecommerce_data: Dataset):
    schema = ecommerce_data.schema
    user_id = schema.select_by_tag(Tags.USER_ID)
    item_id = schema.select_by_tag(Tags.ITEM_ID)

    # TODO: Change this for new RetrievalModel
    encoders = mm.SequentialBlock(
        mm.ParallelBlock(
            mm.EmbeddingTable(64, user_id.first), mm.EmbeddingTable(64, item_id.first)
        ),
        RenameFeatures(dict(user_id="query", item_id="candidate")),
    )

    mf = mm.Model(encoders, mm.ContrastiveOutput(item_id, "in-batch"))

    testing_utils.model_test(mf, ecommerce_data, run_eagerly=True)


def test_constrastive_mf_weights_in_output(ecommerce_data: Dataset):
    schema = ecommerce_data.schema
    schema["item_id"] = schema["item_id"].with_tags([Tags.TARGET])
    user_id = schema.select_by_tag(Tags.USER_ID)
    item_id = schema.select_by_tag(Tags.ITEM_ID)

    # TODO: Change this for new RetrievalModel
    encoder = mm.TabularBlock(mm.EmbeddingTable(64, user_id.first), aggregation="concat")

    mf = mm.Model(encoder, mm.ContrastiveOutput(item_id, "in-batch"))

    testing_utils.model_test(mf, ecommerce_data, run_eagerly=True)


def test_two_tower_constrastive(ecommerce_data: Dataset):
    model = mm.RetrievalModel(
        mm.TwoTowerBlock(ecommerce_data.schema, query_tower=mm.MLPBlock([8])),
        mm.ContrastiveOutput(
            ecommerce_data.schema.select_by_tag(Tags.ITEM_ID),
            negative_samplers="in-batch",
            candidate_name="item",
        ),
    )

    testing_utils.model_test(model, ecommerce_data)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_contrastive_output(ecommerce_data: Dataset, run_eagerly):
    schema = ecommerce_data.schema
    schema["item_category"] = schema["item_category"].with_tags(
        schema["item_category"].tags + "target"
    )
    ecommerce_data.schema = schema
    model = mm.Model(
        mm.InputBlock(schema),
        mm.MLPBlock([8]),
        mm.ContrastiveOutput(
            schema["item_category"],
            negative_samplers=PopularityBasedSamplerV2(max_id=100, max_num_samples=20),
        ),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "ndcg_at_10",
        "precision_at_10",
        "map_at_10",
        "mrr_at_10",
        "recall_at_10",
        "regularization_loss",
    }


def test_setting_negative_sampling_strategy(sequence_testing_data: Dataset):
    dataloader, schema = _next_item_loader(sequence_testing_data)
    model = mm.Model(
        mm.InputBlockV2(schema),
        mm.MLPBlock([32]),
        mm.ContrastiveOutput(prediction=schema["item_id_seq"]),
    )
    batch = next(iter(dataloader))
    output = model(batch[0], batch[1], training=True)
    assert output.shape == (batch[1].shape[0], 51997)

    model.compile(
        optimizer="adam",
        negative_sampling=[PopularityBasedSamplerV2(max_id=51996, max_num_samples=20)],
    )

    output = model(batch[0], batch[1], training=True)
    assert output.outputs.shape == (batch[1].shape[0], 21)

    model.compile(
        optimizer="adam",
        negative_sampling=["in-batch", PopularityBasedSamplerV2(max_id=51996, max_num_samples=20)],
    )
    output = model(batch[0], batch[1], training=True)
    assert output.outputs.shape == (batch[1].shape[0], 71)


def _next_item_loader(sequence_testing_data: Dataset):
    def _last_interaction_as_target(inputs, targets):
        inputs = mm.AsRaggedFeatures()(inputs)
        items = inputs["item_id_seq"]
        _items = items[:, :-1]
        targets = tf.one_hot(items[:, -1:].flat_values, 51997)
        inputs["item_id_seq"] = _items
        return inputs, targets

    schema = sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    sequence_testing_data.schema = schema
    dataloader = BatchedDataset(sequence_testing_data, batch_size=50)
    dataloader = dataloader.map(_last_interaction_as_target)
    return dataloader, schema
