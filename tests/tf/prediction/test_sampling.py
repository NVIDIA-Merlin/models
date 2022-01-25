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


def test_inbatch_sampler():
    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=10000, dtype=tf.int32)

    inbatch_sampler = ml.InBatchSampler(batch_size=10)

    input_data = ml.ItemSamplerData(item_embeddings, {"item_id": item_ids})
    output_data = inbatch_sampler(input_data.__dict__)

    tf.assert_equal(input_data.items_embeddings, output_data.items_embeddings)
    for feat_name in output_data.items_metadata:
        tf.assert_equal(input_data.items_metadata[feat_name], output_data.items_metadata[feat_name])


def test_inbatch_sampler_no_metadata_features():
    embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)

    inbatch_sampler = ml.InBatchSampler(batch_size=10)

    input_data = ml.ItemSamplerData(items_embeddings=embeddings, items_metadata={})
    output_data = inbatch_sampler(input_data.__dict__)

    tf.assert_equal(input_data.items_embeddings, output_data.items_embeddings)
    assert output_data.items_metadata == {}


def test_inbatch_sampler_metadata_diff_shape():
    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(11,), minval=1, maxval=10000, dtype=tf.int32)

    inbatch_sampler = ml.InBatchSampler(batch_size=10)

    input_data = ml.ItemSamplerData(item_embeddings, {"item_id": item_ids})

    with pytest.raises(Exception) as excinfo:
        _ = inbatch_sampler(input_data.__dict__)
    assert "The batch size (first dim) of items_embeddings" in str(excinfo.value)


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_cached_batches_sampler_add_sample_single_batch(ignore_last_batch_on_sample):
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        batch_size=10,
        num_batches_to_cache=2,
        ignore_last_batch_on_sample=ignore_last_batch_on_sample,
    )

    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=10000, dtype=tf.int32)
    item_image_embeddings = tf.random.uniform(shape=(10, 8), dtype=tf.float32)

    input_data = ml.ItemSamplerData(
        item_embeddings, {"item_id": item_ids, "image": item_image_embeddings}
    )
    output_data = cached_batches_sampler(input_data.__dict__)

    if ignore_last_batch_on_sample:
        tf.assert_equal(tf.shape(output_data.items_embeddings)[0], 0)
        tf.assert_equal(tf.shape(output_data.items_metadata["item_id"])[0], 0)
        tf.assert_equal(tf.shape(output_data.items_metadata["image"])[0], 0)
    else:
        tf.assert_equal(output_data.items_embeddings, item_embeddings)
        tf.assert_equal(output_data.items_metadata["item_id"], item_ids)
        tf.assert_equal(output_data.items_metadata["image"], item_image_embeddings)


def test_cached_batches_sampler_add_sample_only_item_embedding():
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        batch_size=10,
        num_batches_to_cache=2,
        ignore_last_batch_on_sample=False,
    )

    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)

    input_data = ml.ItemSamplerData(item_embeddings, {})
    output_data = cached_batches_sampler(input_data.__dict__)

    tf.assert_equal(output_data.items_embeddings, item_embeddings)
    assert output_data.items_metadata == {}


def test_cached_batches_sampler_add_diff_item_emb_metadata():
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        batch_size=10,
        num_batches_to_cache=2,
        ignore_last_batch_on_sample=False,
    )

    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(11,), minval=1, maxval=10000, dtype=tf.int32)

    input_data = ml.ItemSamplerData(item_embeddings, {"item_id": item_ids})

    with pytest.raises(Exception) as excinfo:
        _ = cached_batches_sampler(input_data.__dict__)
    assert "The batch size (first dim) of items_embeddings" in str(excinfo.value)


def test_cached_batches_sampler_trying_add_sample_before_built():
    cached_batches_sampler = ml.CachedCrossBatchSampler(
        batch_size=10,
        num_batches_to_cache=2,
        ignore_last_batch_on_sample=False,
    )

    with pytest.raises(Exception) as excinfo:
        input_data = ml.ItemSamplerData(items_embeddings=None, items_metadata={})
        cached_batches_sampler.add(input_data)
    assert "The CachedCrossBatchSampler layer was not built yet." in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        input_data = ml.ItemSamplerData(items_embeddings=None, items_metadata={})
        _ = cached_batches_sampler.sample()
    assert "The CachedCrossBatchSampler layer was not built yet." in str(excinfo.value)


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_cached_batches_sampler_multiple_batches(ignore_last_batch_on_sample):
    batch_size = 10
    num_batches_to_cache = 2
    num_expected_samples = batch_size * num_batches_to_cache

    cached_batches_sampler = ml.CachedCrossBatchSampler(
        batch_size=batch_size,
        num_batches_to_cache=num_batches_to_cache,
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

        input_data = ml.ItemSamplerData(item_embeddings, {"item_id": item_ids})
        output_data = cached_batches_sampler(input_data.__dict__)
        expected_sampled_items = min(
            batch_size * (step - 1 if ignore_last_batch_on_sample else step), num_expected_samples
        )

        # Checks if the number of sampled items matches the expected
        tf.assert_equal(tf.shape(output_data.items_embeddings)[0], expected_sampled_items)
        for feat_name in output_data.items_metadata:
            tf.assert_equal(
                tf.shape(output_data.items_metadata[feat_name])[0], expected_sampled_items
            )

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

    tf.assert_equal(output_data.items_embeddings, expected_sampled_item_embeddings)
    tf.assert_equal(output_data.items_metadata["item_id"], expected_sampled_item_ids)


@pytest.mark.parametrize("ignore_last_batch_on_sample", [True, False])
def test_cached_batches_sampler_max_num_samples(ignore_last_batch_on_sample):
    batch_size = 10
    num_batches_to_cache = 2

    cached_batches_sampler = ml.CachedCrossBatchSampler(
        batch_size=batch_size,
        num_batches_to_cache=num_batches_to_cache,
        ignore_last_batch_on_sample=ignore_last_batch_on_sample,
    )
    assert cached_batches_sampler.max_num_samples == batch_size * num_batches_to_cache


def test_cached_uniform_sampler():
    queue_capacity = 15
    uniform_sampler = ml.CachedUniformSampler(
        capacity=queue_capacity,
    )

    # Adding 4 new items ids but two of them with repeated item ids (should add only
    # the 2 unique ones)
    item_ids0 = tf.constant([0, 0, 1, 1], dtype=tf.int32)
    item_embeddings0 = tf.random.uniform(shape=(4, 5), dtype=tf.float32)
    input_data_0 = ml.ItemSamplerData(item_embeddings0, {"item_id": item_ids0})
    output_data_0 = uniform_sampler(input_data_0.__dict__)
    assert tuple(output_data_0.items_embeddings.shape) == (2, 5)
    tf.assert_equal(output_data_0.items_metadata["item_id"], [0, 1])
    tf.assert_equal(
        output_data_0.items_embeddings,
        tf.stack([item_embeddings0[0], item_embeddings0[2]], axis=0),
    )

    # Updating two existing items (0,1) and adding 8 new items ids (2-9)
    item_ids1 = tf.range(0, 10)
    item_embeddings1 = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    input_data_1 = ml.ItemSamplerData(item_embeddings1, {"item_id": item_ids1})
    output_data_1 = uniform_sampler(input_data_1.__dict__)
    assert tuple(output_data_1.items_embeddings.shape) == (10, 5)
    tf.assert_equal(output_data_1.items_metadata["item_id"], item_ids1)
    tf.assert_equal(output_data_1.items_embeddings, item_embeddings1)

    # Updating existing items ids (7,8,9) and adding 4 new items ids (10-12)
    item_ids2 = tf.range(7, 13)
    item_embeddings2 = tf.random.uniform(shape=(6, 5), dtype=tf.float32)
    input_data_2 = ml.ItemSamplerData(item_embeddings2, {"item_id": item_ids2})
    output_data_2 = uniform_sampler(input_data_2.__dict__)
    assert tuple(output_data_2.items_embeddings.shape) == (13, 5)
    tf.assert_equal(
        output_data_2.items_metadata["item_id"], tf.concat([item_ids1[:7], item_ids2], axis=0)
    )
    tf.assert_equal(
        output_data_2.items_embeddings, tf.concat([item_embeddings1[:7], item_embeddings2], axis=0)
    )

    # Updating existing items ids (0,1,2)
    item_ids3 = tf.range(0, 3)
    item_embeddings3 = tf.random.uniform(shape=(3, 5), dtype=tf.float32)
    input_data_3 = ml.ItemSamplerData(item_embeddings3, {"item_id": item_ids3})
    output_data_3 = uniform_sampler(input_data_3.__dict__)
    assert tuple(output_data_3.items_embeddings.shape) == (13, 5)
    tf.assert_equal(
        output_data_3.items_metadata["item_id"],
        tf.concat([item_ids3[0:3], item_ids1[3:7], item_ids2], axis=0),
    )
    tf.assert_equal(
        output_data_3.items_embeddings,
        tf.concat([item_embeddings3[0:3], item_embeddings1[3:7], item_embeddings2], axis=0),
    )

    # Adding four new items (13,14,15,16). As adding those items will exceed the queue capacity,
    # the first 2 items added to the queue will be removed to keep the queue size = 15
    item_ids4 = tf.range(13, 17)
    item_embeddings4 = tf.random.uniform(shape=(4, 5), dtype=tf.float32)
    input_data_4 = ml.ItemSamplerData(item_embeddings4, {"item_id": item_ids4})
    output_data_4 = uniform_sampler(input_data_4.__dict__)
    assert tuple(output_data_4.items_embeddings.shape) == (queue_capacity, 5)
    tf.assert_equal(
        output_data_4.items_metadata["item_id"],
        tf.concat([item_ids3[2:3], item_ids1[3:7], item_ids2, item_ids4], axis=0),
    )
    tf.assert_equal(
        output_data_4.items_embeddings,
        tf.concat(
            [item_embeddings3[2:3], item_embeddings1[3:7], item_embeddings2, item_embeddings4],
            axis=0,
        ),
    )


def test_cached_uniform_sampler_no_item_id():
    uniform_sampler = ml.CachedUniformSampler(
        capacity=10,
    )
    item_embeddings = tf.random.uniform(shape=(4, 5), dtype=tf.float32)
    input_data_0 = ml.ItemSamplerData(item_embeddings, items_metadata={})

    with pytest.raises(AssertionError) as excinfo:
        _ = uniform_sampler(input_data_0.__dict__)
    assert "The 'item_id' metadata feature is required by UniformSampler" in str(excinfo.value)


def test_cached_uniform_sampler_trying_add_sample_before_built():
    cached_batches_sampler = ml.CachedUniformSampler(
        capacity=10,
    )

    with pytest.raises(Exception) as excinfo:
        input_data = ml.ItemSamplerData(items_embeddings=None, items_metadata={})
        cached_batches_sampler.add(input_data)
    assert "The CachedUniformSampler layer was not built yet." in str(excinfo.value)

    with pytest.raises(Exception) as excinfo:
        input_data = ml.ItemSamplerData(items_embeddings=None, items_metadata={})
        _ = cached_batches_sampler.sample()
    assert "The CachedUniformSampler layer was not built yet." in str(excinfo.value)
