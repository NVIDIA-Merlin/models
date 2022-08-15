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

from merlin.models.tf.blocks.retrieval.base import ItemRetrievalScorer
from merlin.models.tf.blocks.sampling.in_batch import InBatchSampler


def test_item_retrieval_scorer_reload():
    samplers = [InBatchSampler()]
    sampling_downscore_false_negatives = False
    sampling_downscore_false_negatives_value = 2.0
    item_id_feature_name = "my_item_id"
    item_domain = "my_item_domain"
    query_name = "my_query_name"
    item_name = "my_item_name"
    cache_query = True
    sampled_softmax_mode = True
    store_negative_ids = True
    item_retrieval_task = ItemRetrievalScorer(
        samplers=samplers,
        sampling_downscore_false_negatives=sampling_downscore_false_negatives,
        sampling_downscore_false_negatives_value=sampling_downscore_false_negatives_value,
        item_id_feature_name=item_id_feature_name,
        item_domain=item_domain,
        query_name=query_name,
        item_name=item_name,
        cache_query=cache_query,
        sampled_softmax_mode=sampled_softmax_mode,
        store_negative_ids=store_negative_ids,
    )
    serialized = tf.keras.layers.serialize(item_retrieval_task)
    reloaded = tf.keras.layers.deserialize(serialized)
    assert len(reloaded.samplers) == len(samplers)
    assert reloaded.downscore_false_negatives == sampling_downscore_false_negatives
    assert reloaded.false_negatives_score == sampling_downscore_false_negatives_value
    assert reloaded.item_id_feature_name == item_id_feature_name
    assert reloaded.item_domain == item_domain
    assert reloaded.query_name == query_name
    assert reloaded.item_name == item_name
    assert reloaded.cache_query == cache_query
    assert reloaded.sampled_softmax_mode == sampled_softmax_mode
    assert reloaded.store_negative_ids == store_negative_ids
