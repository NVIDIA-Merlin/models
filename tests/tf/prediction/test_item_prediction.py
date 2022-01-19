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

import merlin_models.tf as ml
from merlin_models.data.synthetic import SyntheticData
from merlin_standard_lib import Tag

tf = pytest.importorskip("tensorflow")


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


@pytest.mark.parametrize("run_eagerly", [True, False])
@pytest.mark.parametrize("weight_tying", [True, False])
@pytest.mark.parametrize("sampled_softmax", [True, False])
def test_last_item_prediction_task(
    sequence_testing_data: SyntheticData,
    run_eagerly: bool,
    weight_tying: bool,
    sampled_softmax: bool,
):
    inputs = ml.InputBlock(
        sequence_testing_data.schema,
        aggregation="concat",
        seq=False,
        masking="clm",
        split_sparse=True,
    )
    if sampled_softmax:
        loss = tf.nn.softmax_cross_entropy_with_logits
        metrics = ml.ranking_metrics(top_ks=[10, 20], labels_onehot=False)
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ml.ranking_metrics(top_ks=[10, 20], labels_onehot=True)
    task = ml.NextItemPredictionTask(
        schema=sequence_testing_data.schema,
        loss=loss,
        metrics=metrics,
        masking=True,
        weight_tying=weight_tying,
        sampled_softmax=sampled_softmax,
    )

    model = inputs.connect(ml.MLPBlock([64]), task, context=ml.BlockContext())
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    losses = model.fit(sequence_testing_data.tf_dataloader(batch_size=50), epochs=2)

    assert len(losses.epoch) == 2
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list

    out = model(sequence_testing_data.tf_tensor_dict)
    assert out.shape[-1] == 51997


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_youtube_dnn(
    sequence_testing_data: SyntheticData,
    run_eagerly: bool,
):
    model = ml.YoutubeDNN(schema=sequence_testing_data.schema)
    model.compile(optimizer="adam", run_eagerly=run_eagerly)

    losses = model.fit(sequence_testing_data.tf_dataloader(batch_size=50), epochs=2)

    assert len(losses.epoch) == 2
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list
    out = model(sequence_testing_data.tf_tensor_dict)
    assert out.shape[-1] == 51997
