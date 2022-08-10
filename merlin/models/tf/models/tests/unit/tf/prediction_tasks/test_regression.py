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

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_regression_head(ecommerce_data: Dataset, run_eagerly):
    body = ml.InputBlock(ecommerce_data.schema).connect(ml.MLPBlock([64]))
    model = ml.Model(body, ml.RegressionTask("click"))

    testing_utils.model_test(model, ecommerce_data)
