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

import merlin.models.tf as ml
from merlin.io import Dataset


def test_sequential_block_yoochoose(testing_data: Dataset):
    body = ml.InputBlock(testing_data.schema).connect(ml.MLPBlock([64]))

    outputs = body(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert list(outputs.shape) == [100, 64]
