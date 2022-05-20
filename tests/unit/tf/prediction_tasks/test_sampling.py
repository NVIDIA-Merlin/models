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


def test_inbatch_sampler():
    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=10000, dtype=tf.int32)

    inbatch_sampler = ml.InBatchSampler()

    input_data = ml.EmbeddingWithMetadata(item_embeddings, {"item_id": item_ids})
    output_data = inbatch_sampler(input_data.__dict__)

    tf.assert_equal(input_data.embeddings, output_data.embeddings)
    for feat_name in output_data.metadata:
        tf.assert_equal(input_data.metadata[feat_name], output_data.metadata[feat_name])


def test_inbatch_sampler_no_metadata_features():
    embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)

    inbatch_sampler = ml.InBatchSampler()

    input_data = ml.EmbeddingWithMetadata(embeddings=embeddings, metadata={})
    output_data = inbatch_sampler(input_data.__dict__)

    tf.assert_equal(input_data.embeddings, output_data.embeddings)
    assert output_data.metadata == {}


def test_inbatch_sampler_metadata_diff_shape():
    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(11,), minval=1, maxval=10000, dtype=tf.int32)

    inbatch_sampler = ml.InBatchSampler()

    input_data = ml.EmbeddingWithMetadata(item_embeddings, {"item_id": item_ids})

    with pytest.raises(Exception) as excinfo:
        _ = inbatch_sampler(input_data.__dict__)
    assert "The batch size (first dim) of embedding" in str(excinfo.value)


def test_popularity_sampler():
    num_classes = 100
    min_id = 2
    num_sampled = 10
    item_weights = tf.random.uniform(shape=(num_classes, 5), dtype=tf.float32)
    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=num_classes, dtype=tf.int32)

    popularity_sampler = ml.PopularityBasedSampler(
        max_num_samples=num_sampled, max_id=num_classes - 1, min_id=min_id
    )

    input_data = ml.EmbeddingWithMetadata(item_embeddings, {"item_id": item_ids})
    output_data = popularity_sampler(input_data.__dict__, item_weights)

    assert len(tf.unique_with_counts(output_data.metadata["item_id"])[0]) == num_sampled
    tf.assert_equal(
        tf.nn.embedding_lookup(item_weights, output_data.metadata["item_id"]),
        output_data.embeddings,
    )
    tf.assert_equal(tf.reduce_all(output_data.metadata["item_id"] >= min_id), True)


def test_popularity_sampler_no_item_id():
    num_sampled = 10
    num_classes = 10000
    item_weights = tf.random.uniform(shape=(num_classes, 5), dtype=tf.float32)
    popularity_sampler = ml.PopularityBasedSampler(max_num_samples=num_sampled, max_id=num_classes)

    item_embeddings = tf.random.uniform(shape=(2, 5), dtype=tf.float32)
    input_data_0 = ml.EmbeddingWithMetadata(item_embeddings, metadata={})

    with pytest.raises(AssertionError) as excinfo:
        _ = popularity_sampler(input_data_0.__dict__, item_weights)
    assert "The 'item_id' metadata feature is required by PopularityBasedSampler" in str(
        excinfo.value
    )


def test_popularity_sampler_weights_dim_diff_max_num_samples():
    num_sampled = 10
    num_classes = 10000
    item_weights = tf.random.uniform(shape=(50, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=num_classes, dtype=tf.int32)

    popularity_sampler = ml.PopularityBasedSampler(max_num_samples=num_sampled, max_id=num_classes)
    item_embeddings = tf.random.uniform(shape=(2, 5), dtype=tf.float32)
    input_data_0 = ml.EmbeddingWithMetadata(item_embeddings, {"item_id": item_ids})

    with pytest.raises(Exception) as excinfo:
        _ = popularity_sampler(input_data_0.__dict__, item_weights)
    assert "The first dimension of the items embeddings (50)" in str(excinfo.value)


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_cached_batches_sampler_add_sample_single_batch(ignore_last_batch_on_sample):
    batch_size = 10
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * 2,
        ignore_last_batch_on_sample=ignore_last_batch_on_sample,
    )

    item_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(batch_size,), minval=1, maxval=10000, dtype=tf.int32)
    item_image_embeddings = tf.random.uniform(shape=(batch_size, 8), dtype=tf.float32)

    input_data = ml.EmbeddingWithMetadata(
        item_embeddings, {"item_id": item_ids, "image": item_image_embeddings}
    )
    output_data = cached_batches_sampler(input_data.__dict__)

    if ignore_last_batch_on_sample:
        tf.assert_equal(tf.shape(output_data.embeddings)[0], 0)
        tf.assert_equal(tf.shape(output_data.metadata["item_id"])[0], 0)
        tf.assert_equal(tf.shape(output_data.metadata["image"])[0], 0)
    else:
        tf.assert_equal(output_data.embeddings, item_embeddings)
        tf.assert_equal(output_data.metadata["item_id"], item_ids)
        tf.assert_equal(output_data.metadata["image"], item_image_embeddings)


def test_cached_batches_sampler_add_sample_only_item_embedding():
    batch_size = 10
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * 2,
        ignore_last_batch_on_sample=False,
    )

    item_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)

    input_data = ml.EmbeddingWithMetadata(item_embeddings, {})
    output_data = cached_batches_sampler(input_data.__dict__)

    tf.assert_equal(output_data.embeddings, item_embeddings)
    assert output_data.metadata == {}


def test_cached_batches_sampler_add_diff_item_emb_metadata():
    batch_size = 10
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * 2,
        ignore_last_batch_on_sample=False,
    )

    item_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(batch_size + 1,), minval=1, maxval=10000, dtype=tf.int32)

    input_data = ml.EmbeddingWithMetadata(item_embeddings, {"item_id": item_ids})

    with pytest.raises(Exception) as excinfo:
        _ = cached_batches_sampler(input_data.__dict__)
    assert "The batch size (first dim) of embedding" in str(excinfo.value)


def test_cached_batches_sampler_trying_add_sample_before_built():
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=20,
        ignore_last_batch_on_sample=False,
    )

    with pytest.raises(Exception) as excinfo:
        input_data = ml.EmbeddingWithMetadata(embeddings=None, metadata={})
        cached_batches_sampler.add(input_data)
    assert "The CachedCrossBatchSampler layer was not built yet." in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        input_data = ml.EmbeddingWithMetadata(embeddings=None, metadata={})
        _ = cached_batches_sampler.sample()
    assert "The CachedCrossBatchSampler layer was not built yet." in str(excinfo.value)


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_cached_batches_sampler_multiple_batches(ignore_last_batch_on_sample):
    batch_size = 10
    num_batches_to_cache = 2
    num_expected_samples = batch_size * num_batches_to_cache

    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * num_batches_to_cache,
        ignore_last_batch_on_sample=ignore_last_batch_on_sample,
    )
    batches_items_embeddings_list = []
    batches_items_ids_list = []
    for step in range(1, 4):
        # Making last batch incomplete (lower than the batch size)
        num_rows_current_batch = 10 if step < 3 else 7
        item_embeddings = tf.random.uniform(shape=(num_rows_current_batch, 5), dtype=tf.float32)
        batches_items_embeddings_list.append(item_embeddings)
        item_ids = tf.random.uniform(
            shape=(num_rows_current_batch,), minval=1, maxval=10000, dtype=tf.int32
        )
        batches_items_ids_list.append(item_ids)

        input_data = ml.EmbeddingWithMetadata(item_embeddings, {"item_id": item_ids})
        output_data = cached_batches_sampler(input_data.__dict__)
        expected_sampled_items = min(
            batch_size * (step - 1 if ignore_last_batch_on_sample else step), num_expected_samples
        )

        # Checks if the number of sampled items matches the expected
        tf.assert_equal(tf.shape(output_data.embeddings)[0], expected_sampled_items)
        for feat_name in output_data.metadata:
            tf.assert_equal(tf.shape(output_data.metadata[feat_name])[0], expected_sampled_items)

    expected_sampled_item_embeddings = tf.concat(batches_items_embeddings_list, axis=0)
    expected_sampled_item_ids = tf.concat(batches_items_ids_list, axis=0)

    if ignore_last_batch_on_sample:
        expected_sampled_item_embeddings = expected_sampled_item_embeddings[
            :-num_rows_current_batch
        ]
        expected_sampled_item_ids = expected_sampled_item_ids[:-num_rows_current_batch]
    else:
        # Checks if the sampled items correspond to the 2nd and 3rd batches
        expected_sampled_item_embeddings = expected_sampled_item_embeddings[num_rows_current_batch:]
        expected_sampled_item_ids = expected_sampled_item_ids[num_rows_current_batch:]

    tf.assert_equal(output_data.embeddings, expected_sampled_item_embeddings)
    tf.assert_equal(output_data.metadata["item_id"], expected_sampled_item_ids)


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_cached_batches_sampler_max_num_samples(ignore_last_batch_on_sample):
    batch_size = 10
    num_batches_to_cache = 2

    cached_batches_sampler = ml.CachedCrossBatchSampler(
        capacity=batch_size * num_batches_to_cache,
        ignore_last_batch_on_sample=ignore_last_batch_on_sample,
    )
    assert cached_batches_sampler.max_num_samples == batch_size * num_batches_to_cache


def test_cached_uniform_sampler_adds_or_updates_items():
    queue_capacity = 15
    uniform_sampler = ml.CachedUniformSampler(
        capacity=queue_capacity, ignore_last_batch_on_sample=False
    )

    # Adding 4 new items ids but two of them with repeated item ids (should add only
    # the 2 unique ones)
    item_ids0 = tf.constant([0, 0, 1, 1], dtype=tf.int32)
    item_embeddings0 = tf.random.uniform(shape=(4, 5), dtype=tf.float32)
    input_data_0 = ml.EmbeddingWithMetadata(item_embeddings0, {"item_id": item_ids0})
    output_data_0 = uniform_sampler(input_data_0.__dict__)
    assert tuple(output_data_0.embeddings.shape) == (2, 5)
    tf.assert_equal(output_data_0.metadata["item_id"], [0, 1])
    tf.assert_equal(
        output_data_0.embeddings,
        tf.stack([item_embeddings0[0], item_embeddings0[2]], axis=0),
    )

    # Updating two existing items (0,1) and adding 8 new items ids (2-9)
    item_ids1 = tf.range(0, 10)
    item_embeddings1 = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    input_data_1 = ml.EmbeddingWithMetadata(item_embeddings1, {"item_id": item_ids1})
    output_data_1 = uniform_sampler(input_data_1.__dict__)
    assert tuple(output_data_1.embeddings.shape) == (10, 5)
    tf.assert_equal(output_data_1.metadata["item_id"], item_ids1)
    tf.assert_equal(output_data_1.embeddings, item_embeddings1)

    # Updating existing items ids (7,8,9) and adding 4 new items ids (10-12)
    item_ids2 = tf.range(7, 13)
    item_embeddings2 = tf.random.uniform(shape=(6, 5), dtype=tf.float32)
    input_data_2 = ml.EmbeddingWithMetadata(item_embeddings2, {"item_id": item_ids2})
    output_data_2 = uniform_sampler(input_data_2.__dict__)
    assert tuple(output_data_2.embeddings.shape) == (13, 5)
    tf.assert_equal(
        output_data_2.metadata["item_id"], tf.concat([item_ids1[:7], item_ids2], axis=0)
    )
    tf.assert_equal(
        output_data_2.embeddings, tf.concat([item_embeddings1[:7], item_embeddings2], axis=0)
    )

    # Updating existing items ids (0,1,2)
    item_ids3 = tf.range(0, 3)
    item_embeddings3 = tf.random.uniform(shape=(3, 5), dtype=tf.float32)
    input_data_3 = ml.EmbeddingWithMetadata(item_embeddings3, {"item_id": item_ids3})
    output_data_3 = uniform_sampler(input_data_3.__dict__)
    assert tuple(output_data_3.embeddings.shape) == (13, 5)
    tf.assert_equal(
        output_data_3.metadata["item_id"],
        tf.concat([item_ids3[0:3], item_ids1[3:7], item_ids2], axis=0),
    )
    tf.assert_equal(
        output_data_3.embeddings,
        tf.concat([item_embeddings3[0:3], item_embeddings1[3:7], item_embeddings2], axis=0),
    )

    # Adding four new items (13,14,15,16). As adding those items will exceed the queue capacity,
    # the first 2 items added to the queue will be removed to keep the queue size = 15
    item_ids4 = tf.range(13, 17)
    item_embeddings4 = tf.random.uniform(shape=(4, 5), dtype=tf.float32)
    input_data_4 = ml.EmbeddingWithMetadata(item_embeddings4, {"item_id": item_ids4})
    output_data_4 = uniform_sampler(input_data_4.__dict__)
    assert tuple(output_data_4.embeddings.shape) == (queue_capacity, 5)
    tf.assert_equal(
        output_data_4.metadata["item_id"],
        tf.concat([item_ids3[2:3], item_ids1[3:7], item_ids2, item_ids4], axis=0),
    )
    tf.assert_equal(
        output_data_4.embeddings,
        tf.concat(
            [item_embeddings3[2:3], item_embeddings1[3:7], item_embeddings2, item_embeddings4],
            axis=0,
        ),
    )


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_cached_uniform_sampler_expected_number_examples(ignore_last_batch_on_sample):
    queue_capacity = 100
    uniform_sampler = ml.CachedUniformSampler(
        capacity=queue_capacity, ignore_last_batch_on_sample=ignore_last_batch_on_sample
    )

    for step in range(1, 4):
        batch_size = 10
        # Ensures item ids are unique
        item_ids = tf.range(step * batch_size, (step * batch_size) + 10)
        item_embeddings = tf.random.uniform(shape=(batch_size, 5), dtype=tf.float32)
        input_data = ml.EmbeddingWithMetadata(item_embeddings, {"item_id": item_ids})
        output_data = uniform_sampler(input_data.__dict__)

        expected_samples = batch_size * (step - 1 if ignore_last_batch_on_sample else step)
        assert tuple(output_data.embeddings.shape) == (expected_samples, 5)
        assert tuple(output_data.metadata["item_id"].shape) == (expected_samples,)


def test_cached_uniform_sampler_no_item_id():
    uniform_sampler = ml.CachedUniformSampler(
        capacity=10,
    )
    item_embeddings = tf.random.uniform(shape=(4, 5), dtype=tf.float32)
    input_data_0 = ml.EmbeddingWithMetadata(item_embeddings, metadata={})

    with pytest.raises(AssertionError) as excinfo:
        _ = uniform_sampler(input_data_0.__dict__)
    assert "The 'item_id' metadata feature is required by UniformSampler" in str(excinfo.value)


def test_cached_uniform_sampler_trying_add_sample_before_built():
    cached_batches_sampler = ml.CachedUniformSampler(
        capacity=10,
    )

    with pytest.raises(Exception) as excinfo:
        input_data = ml.EmbeddingWithMetadata(embeddings=None, metadata={})
        cached_batches_sampler.add(input_data.__dict__)
    assert "The CachedUniformSampler layer was not built yet." in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        input_data = ml.EmbeddingWithMetadata(embeddings=None, metadata={})
        _ = cached_batches_sampler.sample()
    assert "The CachedUniformSampler layer was not built yet." in str(excinfo.value)
