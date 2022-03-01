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

import merlin.models.tf as ml
from merlin.models.data.synthetic import SyntheticData
from merlin.schema import Tags


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_item_retrieval_scorer(ignore_last_batch_on_sample):
    batch_size = 10
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * 2,
        ignore_last_batch_on_sample=ignore_last_batch_on_sample,
    )
    inbatch_sampler = ml.InBatchSampler()

    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[cached_batches_sampler, inbatch_sampler], sampling_downscore_false_negatives=False
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    # First batch
    _, output_scores1 = item_retrieval_scorer.call_targets(
        {"query": users_embeddings, "item": items_embeddings}, {}
    )
    expected_num_samples_inbatch = batch_size
    expected_num_samples_cached = 0 if ignore_last_batch_on_sample else batch_size
    tf.assert_equal(tf.shape(output_scores1)[0], batch_size)
    # Number of negatives plus one positive
    tf.assert_equal(
        tf.shape(output_scores1)[1], expected_num_samples_inbatch + expected_num_samples_cached + 1
    )

    # Second batch
    _, output_scores2 = item_retrieval_scorer.call_targets(
        {"query": users_embeddings, "item": items_embeddings}, {}
    )
    expected_num_samples_cached += batch_size
    tf.assert_equal(tf.shape(output_scores2)[0], batch_size)
    # Number of negatives plus one positive
    tf.assert_equal(
        tf.shape(output_scores2)[1], expected_num_samples_inbatch + expected_num_samples_cached + 1
    )


""" @tf.funcion convert `call_targets` of the ItemRetrievalScorer to graph ops,
In graph-model the exception is not raised.
or this test, we need to be able track the exceptions in graph mode.

def test_item_retrieval_scorer_cached_sampler_no_result_first_batch():
    batch_size = 10

    # CachedCrossBatchSampler is the only sampler here and with ignore_last_batch_on_sample=True
    # for the first batch no sample will be returned, which should raise an exception
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * 2,
        ignore_last_batch_on_sample=True,
    )

    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[cached_batches_sampler], sampling_downscore_false_negatives=False
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    with pytest.raises(Exception) as excinfo:
        _ = item_retrieval_scorer({"query": users_embeddings, "item": items_embeddings}, {})
    assert "No negative items where sampled from samplers" in str(excinfo.value)
"""


def test_item_retrieval_scorer_no_sampler():
    with pytest.raises(Exception) as excinfo:
        users_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
        items_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
        item_retrieval_scorer = ml.ItemRetrievalScorer(
            samplers=[], sampling_downscore_false_negatives=False
        )
        item_retrieval_scorer.call_targets(
            {"query": users_embeddings, "item": items_embeddings}, {}
        )
    assert "At least one sampler is required by ItemRetrievalScorer for negative sampling" in str(
        excinfo.value
    )


def test_item_retrieval_scorer_cached_sampler_downscore_false_negatives_no_item_id_context():
    batch_size = 10

    # CachedCrossBatchSampler is the only sampler here and with ignore_last_batch_on_sample=True
    # for the first batch no sample will be returned, which should raise an exception
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * 2,
        ignore_last_batch_on_sample=False,
    )

    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[cached_batches_sampler], sampling_downscore_false_negatives=True
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    with pytest.raises(Exception) as excinfo:
        _ = item_retrieval_scorer.call_targets(
            {"query": users_embeddings, "item": items_embeddings}, targets={}
        )
    assert "The following required context features should be available for the samplers" in str(
        excinfo.value
    )


def test_item_retrieval_scorer_downscore_false_negatives():
    batch_size = 10

    cached_batches_sampler = ml.InBatchSampler()

    # Adding item id to the context
    item_ids = tf.random.uniform(shape=(batch_size,), minval=1, maxval=10000000, dtype=tf.int32)
    context = ml.BlockContext(feature_names=["item_id"], feature_dtypes={"item_id": tf.int32})
    _ = context({"item_id": item_ids})

    FALSE_NEGATIVE_SCORE = -100_000_000.0
    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[cached_batches_sampler],
        sampling_downscore_false_negatives=True,
        sampling_downscore_false_negatives_value=FALSE_NEGATIVE_SCORE,
        context=context,
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    _, output_scores = item_retrieval_scorer.call_targets(
        {"query": users_embeddings, "item": items_embeddings}, targets={}
    )

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


def test_item_retrieval_scorer_only_positive_when_not_training():
    batch_size = 10

    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[ml.InBatchSampler()],
        sampling_downscore_false_negatives=False,
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    # Calls with training=False, so that only the positive item is scored
    output_scores = item_retrieval_scorer(
        {"query": users_embeddings, "item": items_embeddings}, training=False
    )
    tf.assert_equal(
        (int(tf.shape(output_scores)[0]), int(tf.shape(output_scores)[1])), (batch_size, 1)
    )


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_retrieval_task_inbatch_cached_samplers(
    music_streaming_data: SyntheticData, run_eagerly, ignore_last_batch_on_sample
):
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)
    two_tower = ml.TwoTowerBlock(music_streaming_data.schema, query_tower=ml.MLPBlock([512, 256]))

    batch_size = music_streaming_data.tf_tensor_dict["item_id"].shape[0]
    assert batch_size == 100

    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * 2,
        ignore_last_batch_on_sample=ignore_last_batch_on_sample,
    )
    inbatch_sampler = ml.InBatchSampler()

    samplers = [inbatch_sampler, cached_batches_sampler]

    model = two_tower.connect(
        ml.ItemRetrievalTask(
            music_streaming_data._schema, softmax_temperature=2, samplers=samplers, loss="bpr"
        )
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    for batch_step in range(1, 4):
        output = model(music_streaming_data.tf_tensor_dict, training=True)
        _, output = model.loss_block.pre.call_targets(output, targets={}, training=True)
        expected_num_samples_inbatch = batch_size
        expected_num_samples_cached = min(
            batch_size * (batch_step - 1 if ignore_last_batch_on_sample else batch_step),
            cached_batches_sampler.max_num_samples,
        )
        tf.assert_equal(tf.shape(output)[0], batch_size)
        # Number of negatives plus one positive
        tf.assert_equal(
            tf.shape(output)[1], expected_num_samples_inbatch + expected_num_samples_cached + 1
        )


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_retrieval_task_inbatch_cached_samplers_fit(
    music_streaming_data: SyntheticData, run_eagerly, num_epochs=2
):
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)
    two_tower = ml.TwoTowerBlock(music_streaming_data.schema, query_tower=ml.MLPBlock([512, 256]))

    batch_size = 100

    samplers = [
        ml.InBatchSampler(),
        ml.CachedCrossBatchSampler(
            capacity=batch_size * 3,
            ignore_last_batch_on_sample=True,
        ),
        ml.CachedUniformSampler(
            capacity=batch_size * 3,
            ignore_last_batch_on_sample=False,
        ),
    ]

    model = two_tower.connect(
        ml.ItemRetrievalTask(
            music_streaming_data._schema, softmax_temperature=2, samplers=samplers, loss="bpr"
        )
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data.tf_dataloader(batch_size=batch_size), epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("weight_tying", [True, False])
@pytest.mark.parametrize("sampled_softmax", [True, False])
def test_last_item_prediction_task(
    sequence_testing_data: SyntheticData,
    run_eagerly: bool,
    weight_tying: bool,
    sampled_softmax: bool,
):
    inputs = ml.InputBlock(
        sequence_testing_data.schema,
        aggregation="concat",
        seq=False,
        masking="clm",
        split_sparse=True,
    )
    if sampled_softmax:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = ml.ranking_metrics(top_ks=[10, 20], labels_onehot=False)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ml.ranking_metrics(top_ks=[10, 20], labels_onehot=True)
    task = ml.NextItemPredictionTask(
        schema=sequence_testing_data.schema,
        loss=loss,
        metrics=metrics,
        masking=True,
        weight_tying=weight_tying,
        sampled_softmax=sampled_softmax,
    )

    model = inputs.connect(ml.MLPBlock([64]), task)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    losses = model.fit(sequence_testing_data.tf_dataloader(batch_size=50), epochs=2)

    assert len(losses.epoch) == 2
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list

    out = model(sequence_testing_data.tf_tensor_dict)
    assert out.shape[-1] == 51997


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_retrieval_task_inbatch_default_sampler(
    music_streaming_data: SyntheticData, run_eagerly, ignore_last_batch_on_sample
):
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)
    two_tower = ml.TwoTowerBlock(music_streaming_data.schema, query_tower=ml.MLPBlock([512, 256]))

    batch_size = music_streaming_data.tf_tensor_dict["item_id"].shape[0]
    assert batch_size == 100

    model = two_tower.connect(
        ml.ItemRetrievalTask(music_streaming_data.schema, softmax_temperature=2, loss="bpr")
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    for _ in range(1, 4):
        output = model(music_streaming_data.tf_tensor_dict, training=True)
        _, output = model.loss_block.pre.call_targets(output, targets={}, training=True)
        expected_num_samples_inbatch = batch_size

        tf.assert_equal(tf.shape(output)[0], batch_size)
        # Number of negatives plus one positive
        tf.assert_equal(tf.shape(output)[1], expected_num_samples_inbatch + 1)
