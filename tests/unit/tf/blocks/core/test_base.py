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
from merlin.io.dataset import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_sequential_block_yoochoose(testing_data: Dataset):
    body = ml.InputBlock(testing_data.schema).connect(ml.MLPBlock([64]))

    outputs = body(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 64]


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class DummyFeaturesBlock(ml.Block):
    def call(self, inputs, feature_context: ml.FeatureContext, **kwargs):
        items = list(feature_context.features.select_by_tag(Tags.ITEM_ID).values.values())[0]
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


@pytest.mark.parametrize("run_eagerly", [True])
def test_block_context_model(ecommerce_data: Dataset, run_eagerly: bool):
    model = ml.Model(
        ml.InputBlock(ecommerce_data.schema),
        ml.MLPBlock([64]),
        DummyFeaturesBlock(),
        ml.BinaryClassificationTask("click"),
    )

    copy_model, _ = testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)

    assert copy_model.context == copy_model.block.layers[0].context
