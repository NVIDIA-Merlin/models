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
from merlin.models.tf.outputs.sampling.popularity import PopularityBasedSamplerV2
from merlin.models.tf.transforms.bias import PopularityLogitsCorrection
from merlin.models.tf.transforms.features import Rename
from merlin.models.tf.utils import testing_utils
from merlin.models.utils import schema_utils
from merlin.schema import Tags


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_contrastive_mf(ecommerce_data: Dataset, run_eagerly: bool):
    schema = ecommerce_data.schema
    user_id = schema.select_by_tag(Tags.USER_ID)
    item_id = schema.select_by_tag(Tags.ITEM_ID)

    # TODO: Change this for new RetrievalModel
    encoders = mm.SequentialBlock(
        mm.ParallelBlock(
            mm.EmbeddingTable(64, user_id.first), mm.EmbeddingTable(64, item_id.first)
        ),
        Rename(dict(user_id="query", item_id="candidate")),
    )

    mf = mm.Model(encoders, mm.ContrastiveOutput(item_id, "in-batch"))

    testing_utils.model_test(mf, ecommerce_data, run_eagerly=run_eagerly, reload_model=True)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_constrastive_mf_weights_in_output(ecommerce_data: Dataset, run_eagerly: bool):
    schema = ecommerce_data.schema
    schema["item_id"] = schema["item_id"].with_tags([Tags.TARGET])
    user_id = schema.select_by_tag(Tags.USER_ID)
    item_id = schema.select_by_tag(Tags.ITEM_ID)

    # TODO: Change this for new RetrievalModel
    encoder = mm.TabularBlock(mm.EmbeddingTable(64, user_id.first), aggregation="concat")

    mf = mm.Model(encoder, mm.ContrastiveOutput(item_id, "in-batch"))

    testing_utils.model_test(mf, ecommerce_data, run_eagerly=run_eagerly, reload_model=True)


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


def test_two_tower_constrastive_with_logq_correction(ecommerce_data: Dataset):
    cardinalities = schema_utils.categorical_cardinalities(ecommerce_data.schema)
    item_id_cardinalities = cardinalities[
        ecommerce_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    ]
    items_frequencies = tf.sort(
        tf.random.uniform((item_id_cardinalities,), minval=0, maxval=1000, dtype=tf.int32)
    )
    post_logits = PopularityLogitsCorrection(
        items_frequencies,
        schema=ecommerce_data.schema,
    )

    model = mm.RetrievalModel(
        mm.TwoTowerBlock(ecommerce_data.schema, query_tower=mm.MLPBlock([8])),
        mm.ContrastiveOutput(
            ecommerce_data.schema.select_by_tag(Tags.ITEM_ID),
            negative_samplers="in-batch",
            candidate_name="item",
            store_negative_ids=True,
            post=post_logits,
        ),
    )

    testing_utils.model_test(model, ecommerce_data, reload_model=True)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_contrastive_output_with_sampled_softmax(ecommerce_data: Dataset, run_eagerly):
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
            negative_samplers=PopularityBasedSamplerV2(max_id=100, max_num_samples=20, min_id=1),
            logq_sampling_correction=True,
        ),
    )

    _, history = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert set(history.history.keys()) == {
        "loss",
        "loss_batch",
        "ndcg_at_10",
        "precision_at_10",
        "map_at_10",
        "mrr_at_10",
        "recall_at_10",
        "regularization_loss",
    }


def test_setting_negative_sampling_strategy(sequence_testing_data: Dataset):
    dataloader, schema = testing_utils.loader_for_last_item_prediction(
        sequence_testing_data, to_one_hot=False
    )
    model_out = mm.ContrastiveOutput(schema["item_id_seq"], "in-batch")
    model = mm.Model(mm.InputBlockV2(schema), mm.MLPBlock([32]), model_out)
    model.compile(optimizer="adam")

    batch = next(iter(dataloader))
    output = model(batch[0], batch[1], training=True)
    assert output[1].shape == (batch[1].shape[0], 51)

    model_out.set_negative_samplers(
        [PopularityBasedSamplerV2(max_id=51996, max_num_samples=20)],
    )

    output = model(batch[0], batch[1], training=True)
    assert output.outputs.shape == (batch[1].shape[0], 21)

    model_out.set_negative_samplers(
        ["in-batch", PopularityBasedSamplerV2(max_id=51996, max_num_samples=20)],
    )
    output = model(batch[0], batch[1], training=True)
    assert output.outputs.shape == (batch[1].shape[0], 71)


def test_contrastive_output_without_sampler(ecommerce_data: Dataset):
    with pytest.raises(Exception) as excinfo:
        inputs, features = _retrieval_inputs_(batch_size=10)
        retrieval_scorer = mm.ContrastiveOutput(
            schema=ecommerce_data.schema, negative_samplers=[], downscore_false_negatives=False
        )
        _ = retrieval_scorer(inputs, features=features, training=True)
        assert (
            "At least one sampler is required by ContrastiveDotProduct for negative sampling"
            in str(excinfo.value)
        )


def test_downscore_false_negatives(ecommerce_data: Dataset):
    batch_size = 10

    inbatch_sampler = mm.InBatchSamplerV2()
    inputs, features = _retrieval_inputs_(batch_size=batch_size)

    FALSE_NEGATIVE_SCORE = -100_000_000.0
    contrastive = mm.ContrastiveOutput(
        ecommerce_data.schema.select_by_tag(Tags.ITEM_ID),
        negative_samplers=[inbatch_sampler],
        downscore_false_negatives=True,
        false_negative_score=FALSE_NEGATIVE_SCORE,
    )

    outputs = contrastive(
        inputs,
        training=True,
        features=features,
    )
    output_scores = outputs.predictions

    output_neg_scores = output_scores[:, 1:]

    diag_mask = tf.eye(tf.shape(output_neg_scores)[0], dtype=tf.bool)
    tf.assert_equal(output_neg_scores[diag_mask], FALSE_NEGATIVE_SCORE)
    tf.assert_equal(
        tf.reduce_all(
            tf.not_equal(
                output_neg_scores[tf.math.logical_not(diag_mask)],
                tf.constant(FALSE_NEGATIVE_SCORE, dtype=output_neg_scores.dtype),
            )
        ),
        True,
    )


def test_contrastive_only_positive_when_not_training(ecommerce_data: Dataset):
    batch_size = 10

    inbatch_sampler = mm.InBatchSamplerV2()
    item_retrieval_prediction = mm.ContrastiveOutput(
        ecommerce_data.schema.select_by_tag(Tags.ITEM_ID),
        negative_samplers=[inbatch_sampler],
        downscore_false_negatives=False,
    )

    inputs, _ = _retrieval_inputs_(batch_size=batch_size)
    output_scores = item_retrieval_prediction(inputs)
    tf.assert_equal(
        (int(tf.shape(output_scores)[0]), int(tf.shape(output_scores)[1])), (batch_size, 1)
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_contrastive_output_with_pairwise_loss(ecommerce_data: Dataset, run_eagerly):
    model = mm.RetrievalModelV2(
        query=mm.Encoder(ecommerce_data.schema.select_by_tag(Tags.USER), mm.MLPBlock([2])),
        candidate=mm.Encoder(ecommerce_data.schema.select_by_tag(Tags.ITEM), mm.MLPBlock([2])),
        output=mm.ContrastiveOutput(
            ecommerce_data.schema.select_by_tag(Tags.ITEM_ID),
            negative_samplers="in-batch",
            candidate_name="item",
        ),
    )
    model.compile(run_eagerly=run_eagerly, loss="bpr-max")
    _ = model.fit(ecommerce_data, batch_size=50, epochs=1)


def _retrieval_inputs_(batch_size):
    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    positive_items = tf.random.uniform(
        shape=(batch_size,), minval=1, maxval=1000000, dtype=tf.int32
    )
    inputs = {"query": users_embeddings, "candidate": items_embeddings}
    features = {"item_id": positive_items, "user_id": None}
    return inputs, features
