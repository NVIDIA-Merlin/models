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

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_regression_head(ecommerce_data: Dataset, run_eagerly: bool):
    body = mm.InputBlock(ecommerce_data.schema).connect(mm.MLPBlock([64]))
    model = mm.Model(body, mm.RegressionTask("click"))

    testing_utils.model_test(model, ecommerce_data, run_eagerly=run_eagerly)


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_regression_head_schema(music_streaming_data: Dataset, run_eagerly: bool):
    body = mm.InputBlock(music_streaming_data.schema).connect(mm.MLPBlock([64]))
    model = mm.Model(body, mm.RegressionTask(music_streaming_data.schema))

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


def test_regression_head_serialization(music_streaming_data: Dataset):
    regression_task = mm.RegressionTask("click")
    assert isinstance(
        regression_task.from_config(regression_task.get_config()), type(regression_task)
    )
