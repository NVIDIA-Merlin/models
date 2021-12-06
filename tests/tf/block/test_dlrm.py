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

tr = pytest.importorskip("merlin_models.tf")


def test_dlrm_block_yoochoose(testing_data: SyntheticData):
    dlrm = tr.DLRMBlock(
        testing_data.schema, bottom_block=tr.MLPBlock([64]), top_block=tr.MLPBlock([64])
    )
    outputs = dlrm(testing_data.tf_tensor_dict)

    assert list(outputs.shape) == [100, 64]
