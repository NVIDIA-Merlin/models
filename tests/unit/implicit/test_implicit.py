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

from merlin.io import Dataset
from merlin.models.implicit import AlternatingLeastSquares
from merlin.schema import Tags


def test_alternating_least_squares(music_streaming_data: Dataset):
    music_streaming_data.schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)

    model = AlternatingLeastSquares(factors=128, iterations=15, regularization=0.01)
    model.fit(music_streaming_data)
    metrics = model.evaluate(music_streaming_data)

    assert all(metric >= 0 for metric in metrics.values())
