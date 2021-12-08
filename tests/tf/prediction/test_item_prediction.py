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


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_retrieval_task(music_streaming_data: SyntheticData, run_eagerly, num_epochs=2):
    music_streaming_data._schema = music_streaming_data.schema.remove_by_tag(Tag.TARGETS)
    two_tower = ml.TwoTowerBlock(music_streaming_data.schema, query_tower=ml.MLPBlock([512, 256]))
    model = two_tower.connect(ml.ItemRetrievalTask(softmax_temperature=2))

    output = model(music_streaming_data.tf_tensor_dict)
    assert output is not None

    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    losses = model.fit(music_streaming_data.tf_dataloader(batch_size=50), epochs=num_epochs)
    assert len(losses.epoch) == num_epochs
    assert all(measure >= 0 for metric in losses.history for measure in losses.history[metric])
