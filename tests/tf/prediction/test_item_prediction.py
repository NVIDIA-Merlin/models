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

from merlin_models.data.synthetic import SyntheticData
from merlin_standard_lib import Tag

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin_models.tf")
test_utils = pytest.importorskip("merlin_models.tf.utils.testing_utils")


def test_retrieval_task(music_streaming_data: SyntheticData, num_epochs=5, run_eagerly=True):

    user_tower = ml.inputs(
        music_streaming_data.schema.select_by_tag(Tag.USER), ml.MLPBlock([512, 256])
    )
    item_tower = ml.inputs(
        music_streaming_data.schema.select_by_tag(Tag.ITEM), ml.MLPBlock([512, 256])
    )

    two_tower = ml.merge({"user": user_tower, "item": item_tower})

    prediction = ml.item_retrieval_task()
    model = two_tower.connect(prediction)

    output = model(music_streaming_data.tf_tensor_dict)
    assert output is not None
    model.compile(optimizer="adam", run_eagerly=True)
    losses = model.fit(music_streaming_data.tf_dataloader(batch_size=50), epochs=4)
    assert len(losses.epoch) == 4
