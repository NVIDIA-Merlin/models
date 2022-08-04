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

import tensorflow as tf

import merlin.models.tf as ml


def test_inbatch_sampler():
    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=10000, dtype=tf.int32)

    inbatch_sampler = ml.InBatchSamplerV2()

    input_data = ml.Items(item_ids, {"item_ids": item_ids}).with_embedding(item_embeddings)
    output_data = inbatch_sampler(input_data)

    tf.assert_equal(input_data.embedding(), output_data.embedding())
    for feat_name in output_data.metadata:
        tf.assert_equal(input_data.metadata[feat_name], output_data.metadata[feat_name])


def test_inbatch_sampler_no_metadata_features():
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=10000, dtype=tf.int32)

    inbatch_sampler = ml.InBatchSamplerV2()

    input_data = ml.Items(item_ids, {})
    output_data = inbatch_sampler(input_data)

    tf.assert_equal(input_data.id, output_data.id)
    assert output_data.metadata == {}
