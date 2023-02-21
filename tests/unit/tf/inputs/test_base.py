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

import merlin.models.tf as mm
from merlin.schema import Tags


def test_concat_sequence(sequence_testing_data):
    sequence_testing_data.schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    seq_schema = sequence_testing_data.schema

    seq_inputs = mm.InputBlockV2(
        seq_schema,
        aggregation="concat",
        categorical=mm.Embeddings(
            seq_schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
        ),
    )

    inputs = mm.sample_batch(sequence_testing_data, 8, include_targets=False, prepare_features=True)

    outputs = seq_inputs(inputs)

    assert outputs.shape.rank == 3
