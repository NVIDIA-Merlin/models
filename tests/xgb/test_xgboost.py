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
import pytest

from merlin.io import Dataset
from merlin.models.xgb import XGBoost


def test_xgboost(music_streaming_data: Dataset):
    objective = "reg:logistic"
    model = XGBoost(objective)
    model.fit(music_streaming_data)
    model.predict(music_streaming_data)
    metrics = model.evaluate(music_streaming_data)

    assert "rmse" in metrics


def test_unsupported_objective(music_streaming_data: Dataset):
    objective = "reg:unknown"
    with pytest.raises(ValueError) as excinfo:
        model = XGBoost(objective)
        model.fit(music_streaming_data)
    assert "Objective not supported" in str(excinfo.value)
