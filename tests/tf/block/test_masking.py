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
from merlin_standard_lib.utils.proto_utils import has_field

ml = pytest.importorskip("merlin_models.tf")


def test_masking_block(sequence_testing_data: SyntheticData):
    list_features = [
        f.name for f in sequence_testing_data.schema.feature if has_field(f, "value_count")
    ]
    schema_list = sequence_testing_data.schema.select_by_name(list_features)
    embedding_block = ml.InputBlock(schema_list, aggregation="concat", seq=True)

    mask_block = ml.CausalLanguageModeling()

    model = embedding_block.connect(mask_block, context=ml.BlockContext())

    batch = sequence_testing_data.tf_tensor_dict
    masked_input = model(batch)
    assert masked_input.shape[-1] == 148


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_masking_head(sequence_testing_data: SyntheticData, run_eagerly: bool):
    list_features = [
        f.name for f in sequence_testing_data.schema.feature if has_field(f, "value_count")
    ]
    schema_list = sequence_testing_data.schema.select_by_name(list_features)

    embedding_block = ml.InputBlock(schema_list, aggregation="concat", seq=True)
    mask_block = ml.CausalLanguageModeling(train_on_last_item_seq_only=True, combiner="mean")
    task = ml.prediction.item_prediction.NextItemPredictionTask(sequence_testing_data.schema)

    model = embedding_block.connect(mask_block, ml.MLPBlock([64]), task, context=ml.BlockContext())
    model.compile(optimizer="adam", run_eagerly=run_eagerly)
    out = model(sequence_testing_data.tf_tensor_dict)
    losses = model.fit(sequence_testing_data.tf_dataloader(batch_size=100), epochs=2)

    assert len(losses.epoch) == 2
    for metric in losses.history.keys():
        assert type(losses.history[metric]) is list
    assert len(losses.history[metric]) == 2
    assert out.shape[-1] == 51997
