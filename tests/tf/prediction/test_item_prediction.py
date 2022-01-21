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

import merlin_models.tf as ml
from merlin_models.data.synthetic import SyntheticData
from merlin_standard_lib import Tag


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_item_retrieval_scorer(ignore_last_batch_on_sample):
    batch_size = 100

    cached_batches_sampler = ml.CachedBatchesSampler(
        num_batches_to_cache=2,
        batch_size=batch_size,
        ignore_last_batch_on_sample=ignore_last_batch_on_sample,
    )
    inbatch_sampler = ml.InBatchSampler(batch_size=batch_size)

    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[cached_batches_sampler, inbatch_sampler], ignore_false_negatives=False
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    # First batch
    output_scores1 = item_retrieval_scorer({"query": users_embeddings, "item": items_embeddings})
    expected_num_samples_inbatch = batch_size
    expected_num_samples_cached = 0 if ignore_last_batch_on_sample else batch_size
    tf.assert_equal(tf.shape(output_scores1)[0], batch_size)
    # Number of negatives plus one positive
    tf.assert_equal(
        tf.shape(output_scores1)[1], expected_num_samples_inbatch + expected_num_samples_cached + 1
    )

    # Second batch
    output_scores2 = item_retrieval_scorer({"query": users_embeddings, "item": items_embeddings})
    expected_num_samples_cached += batch_size
    tf.assert_equal(tf.shape(output_scores2)[0], batch_size)
    # Number of negatives plus one positive
    tf.assert_equal(
        tf.shape(output_scores2)[1], expected_num_samples_inbatch + expected_num_samples_cached + 1
    )


def test_item_retrieval_scorer_cached_sampler_no_result_first_batch():
    batch_size = 100

    # CachedBatchesSampler is the only sampler here and with ignore_last_batch_on_sample=True
    # for the first batch no sample will be returned, which should raise an exception
    cached_batches_sampler = ml.CachedBatchesSampler(
        num_batches_to_cache=2,
        batch_size=batch_size,
        ignore_last_batch_on_sample=True,
    )

    item_retrieval_scorer = ml.ItemRetrievalScorer(
        samplers=[cached_batches_sampler], ignore_false_negatives=False
    )

    users_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    items_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    with pytest.raises(Exception) as excinfo:
        _ = item_retrieval_scorer({"query": users_embeddings, "item": items_embeddings})
    assert "No negative items where sampled from samplers" in str(excinfo.value)


def test_item_retrieval_scorer_no_sampler():
    with pytest.raises(Exception) as excinfo:
        _ = ml.ItemRetrievalScorer(samplers=[], ignore_false_negatives=False)
    assert "At least one sampler is required by ItemRetrievalScorer for negative sampling" in str(
        excinfo.value
    )


# TODO: Add test checking if an exception is raised if ItemRetrievalScorer(...,
# ignore_false_negatives=True) but "item_id" feature is not available in the context


# TODO: Add a test for ItemRetrievalScorer(ignore_false_negatives=True)
# (downscoring false negatives)

# TODO: Add a test for ItemRetrievalScorer(ignore_false_negatives=True) when training=False / True


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_retrieval_task_inbatch_cached_samplers(
    music_streaming_data: SyntheticData, run_eagerly, ignore_last_batch_on_sample
):
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tag.TARGETS)
    two_tower = ml.TwoTowerBlock(music_streaming_data.schema, query_tower=ml.MLPBlock([512, 256]))

    batch_size = music_streaming_data.tf_tensor_dict["item_id"].shape[0]
    assert batch_size == 100

    cached_batches_sampler = ml.CachedBatchesSampler(
        num_batches_to_cache=2,
        batch_size=batch_size,
        ignore_last_batch_on_sample=ignore_last_batch_on_sample,
    )
    inbatch_sampler = ml.InBatchSampler(batch_size=batch_size)

    samplers = [cached_batches_sampler, inbatch_sampler]

    total_sampling_capacity = sum([sampler.max_num_samples for sampler in samplers])
    assert total_sampling_capacity == 300

    model = two_tower.connect(ml.ItemRetrievalTask(softmax_temperature=2, samplers=samplers))

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    for batch_step in range(1, 4):
        output = model(music_streaming_data.tf_tensor_dict, training=True)
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
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tag.TARGETS)
    two_tower = ml.TwoTowerBlock(music_streaming_data.schema, query_tower=ml.MLPBlock([512, 256]))

    batch_size = 100

    samplers = [
        ml.CachedBatchesSampler(
            num_batches_to_cache=3,
            batch_size=batch_size,
            ignore_last_batch_on_sample=True,
        ),
        ml.InBatchSampler(batch_size=batch_size),
    ]

    model = two_tower.connect(ml.ItemRetrievalTask(softmax_temperature=2, samplers=samplers))

    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(music_streaming_data.tf_dataloader(batch_size=batch_size), epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])
