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

tf = pytest.importorskip("tensorflow")
ml = pytest.importorskip("merlin_models.tf")


@pytest.mark.parametrize("mask_block", [ml.CausalLanguageModeling(), ml.MaskedLanguageModeling()])
def test_masking_block(sequence_testing_data: SyntheticData, mask_block):
    list_features = [
        f.name for f in sequence_testing_data.schema.feature if has_field(f, "value_count")
    ]

    schema_list = sequence_testing_data.schema.select_by_name(list_features)
    embedding_block = ml.InputBlock(schema_list, aggregation="concat", seq=True)

    model = embedding_block.connect(mask_block, context=ml.BlockContext())

    batch = sequence_testing_data.tf_tensor_dict
    masked_input = model(batch)
    assert masked_input.shape[-1] == 148
    assert masked_input.shape[1] == 4
