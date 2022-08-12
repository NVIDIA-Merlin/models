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

from merlin.models.tf.blocks.sampling.cross_batch import PopularityBasedSampler


def test_cross_batch_reload():
    max_id = 5
    min_id = 2
    max_num_samples = 4
    seed = 3
    item_id_feature_name = "my_item_id"
    sampler = PopularityBasedSampler(
        max_id,
        min_id=min_id,
        max_num_samples=max_num_samples,
        seed=seed,
        item_id_feature_name=item_id_feature_name,
    )
    serialized = tf.keras.layers.serialize(sampler)
    reloaded = tf.keras.layers.deserialize(serialized)
    assert reloaded.max_id == max_id
    assert reloaded.min_id == min_id
    assert reloaded.max_num_samples == max_num_samples
    assert reloaded.seed == seed
    assert reloaded.item_id_feature_name == item_id_feature_name
