#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

from merlin.models.tf.blocks.sampling.in_batch import InBatchSampler
from merlin.models.tf.prediction_tasks.retrieval import ItemRetrievalTask
from merlin.schema import ColumnSchema, Schema, Tags


def test_item_retrieval_reload():
    schema = Schema([ColumnSchema("item_id", tags=[Tags.CATEGORICAL, Tags.ITEM_ID])])
    item_retrieval_task = ItemRetrievalTask(
        schema, samplers=[InBatchSampler()], logits_temperature=2.0, cache_query=True
    )
    serialized = tf.keras.layers.serialize(item_retrieval_task)
    reloaded = tf.keras.layers.deserialize(serialized)
    assert len(reloaded.samplers) == 1
    assert reloaded.logits_temperature == 2.0
    assert reloaded.cache_query is True
    assert reloaded.schema == schema
