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

from merlin.io import Dataset
from merlin.schema import Tags

import merlin.models.tf as ml
from merlin.models.tf.blocks.sampling.base import Items


def test_inbatch_sampler():
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=10000, dtype=tf.int32)
    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)

    inbatch_sampler = ml.InBatchSampler()

    output_data = inbatch_sampler(Items(item_ids).with_embedding(item_embeddings), training=True)

    tf.assert_equal(item_ids, output_data.ids)
    tf.assert_equal(item_embeddings, output_data.embedding())


def test_inbatch_sampler_no_metadata_features():
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=10000, dtype=tf.int32)

    output_data = ml.InBatchSampler()(Items(item_ids), training=True)

    tf.assert_equal(item_ids, output_data.ids)
    assert output_data.metadata == {}


def test_inbatch_sampler_outside_of_training():
    item_ids = tf.random.uniform(shape=(10,), minval=1, maxval=10000, dtype=tf.int32)

    output_data = ml.InBatchSampler()(Items(item_ids), training=False)

    assert output_data is None


def test_inbatch_sampler_metadata_diff_shape():
    item_ids = tf.random.uniform(shape=(11,), minval=1, maxval=10000, dtype=tf.int32)
    item_embeddings = tf.random.uniform(shape=(10, 5), dtype=tf.float32)

    inbatch_sampler = ml.InBatchSampler()

    with pytest.raises(Exception) as excinfo:
        _ = inbatch_sampler(Items(item_ids).with_embedding(item_embeddings), training=True)
    assert "The batch size (first dim) of embedding" in str(excinfo.value)


def test_inbatch_sampler_in_model(ecommerce_data: Dataset):
    ecommerce_data.schema = ecommerce_data.schema.remove_by_tag(Tags.TARGET)

    model = ml.TwoTowerModel(ecommerce_data.schema, ml.MLPBlock([512, 128]))
    model.compile(
        pre_loss=ml.ContrastiveLearning(ecommerce_data.schema, ml.InBatchSampler()),
        run_eagerly=True,
    )

    model.fit(ecommerce_data, epochs=1, batch_size=100)
