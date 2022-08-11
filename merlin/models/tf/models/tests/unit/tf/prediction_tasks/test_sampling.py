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
