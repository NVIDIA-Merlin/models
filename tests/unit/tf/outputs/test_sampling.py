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
from merlin.models.tf.outputs.sampling.popularity import PopularityBasedSamplerV2


def test_inbatch_sampler():
    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(10, 1), minval=1, maxval=10000, dtype=tf.int32)

    inbatch_sampler = mm.InBatchSamplerV2()

    input_data = mm.Candidate(id=item_ids, metadata={"item_ids": item_ids}).with_embedding(
        item_embeddings
    )
    output_data = inbatch_sampler(input_data)

    tf.assert_equal(input_data.embedding, output_data.embedding)
    for feat_name in output_data.metadata:
        tf.assert_equal(input_data.metadata[feat_name], output_data.metadata[feat_name])


def test_inbatch_sampler_no_metadata_features():
    item_ids = tf.random.uniform(shape=(10, 1), minval=1, maxval=10000, dtype=tf.int32)

    inbatch_sampler = mm.InBatchSamplerV2()

    input_data = mm.Candidate(id=item_ids, metadata={})
    output_data = inbatch_sampler(input_data)

    tf.assert_equal(input_data.id, output_data.id)
    assert output_data.metadata == {}


def test_popularity_sampler():
    num_classes = 1000
    min_id = 2
    num_sampled = 10
    item_ids = tf.random.uniform(shape=(10, 1), minval=1, maxval=num_classes, dtype=tf.int32)

    popularity_sampler = PopularityBasedSamplerV2(
        max_num_samples=num_sampled, max_id=num_classes - 1, min_id=min_id
    )

    input_data = mm.Candidate(id=item_ids, metadata={})
    output_data = popularity_sampler(input_data)

    assert len(tf.unique_with_counts(tf.squeeze(output_data.id))[0]) == num_sampled

    tf.assert_equal(tf.reduce_all(output_data.id >= min_id), True)


def test_popularity_sampler_with_num_samples_greater_than_cardinality():
    num_classes = 50
    min_id = 2
    num_sampled = 100

    with pytest.raises(Exception) as excinfo:
        _ = PopularityBasedSamplerV2(
            max_num_samples=num_sampled, max_id=num_classes - 1, min_id=min_id
        )
    assert "Number of items to sample `100`" in str(excinfo.value)
