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
from typing import List

import pytest
import tensorflow as tf

import merlin.models.tf as ml
from merlin.io.dataset import Dataset
from merlin.schema import Tags


def test_sequential_block_yoochoose(testing_data: Dataset):
    body = ml.InputBlock(testing_data.schema).connect(ml.MLPBlock([64]))

    outputs = body(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 64]


class DummyFeaturesBlock(ml.Block):
    def add_features_to_context(self, feature_shapes) -> List[str]:
        return [Tags.ITEM_ID.value]

    def call(self, inputs, **kwargs):
        items = self.context[Tags.ITEM_ID]
        emb_table = self.context.get_embedding(Tags.ITEM_ID)
        item_embeddings = tf.gather(emb_table, tf.cast(items, tf.int32))
        if tf.rank(item_embeddings) == 3:
            item_embeddings = tf.squeeze(item_embeddings)

        return inputs * item_embeddings

    def compute_output_shape(self, input_shapes):
        return input_shapes

    @property
    def item_embedding_table(self):
        return self.context.get_embedding(Tags.ITEM_ID)


def test_block_context(ecommerce_data: Dataset):
    inputs = ml.InputBlock(ecommerce_data.schema)
    dummy = DummyFeaturesBlock()
    model = inputs.connect(ml.MLPBlock([64]), dummy, context=ml.ModelContext())
    out = model(ml.sample_batch(ecommerce_data, batch_size=100, include_targets=False))

    embeddings = inputs.select_by_name(Tags.CATEGORICAL.value)
    assert (
        dummy.context.get_embedding(Tags.ITEM_ID).shape
        == embeddings.embedding_tables[Tags.ITEM_ID.value].shape
    )

    assert out.shape[-1] == 64


@pytest.mark.parametrize("run_eagerly", [True])
def test_block_context_model(ecommerce_data: Dataset, run_eagerly: bool, tmp_path):
    dummy = DummyFeaturesBlock()
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema),
        ml.MLPBlock([64]),
        dummy,
        ml.BinaryClassificationTask("click"),
    )

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    model.fit(ecommerce_data, batch_size=50, epochs=1)
    model.save(str(tmp_path))

    copy_model = tf.keras.models.load_model(str(tmp_path))
    assert copy_model.context == copy_model.block.layers[0].context
    assert list(copy_model.context._feature_names) == ["item_id"]
    assert len(dict(copy_model.context._feature_dtypes)) == 23

    copy_model.compile(optimizer="adam", run_eagerly=run_eagerly)
    # TODO: Fix prediction-task output name so that we can retrain a model after saving
    # copy_model.fit(ecommerce_data.tf_dataloader(), epochs=1)
