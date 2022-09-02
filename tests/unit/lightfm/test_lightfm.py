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
import numpy as np

from merlin.datasets.synthetic import generate_data
from merlin.models.lightfm import LightFM
from merlin.schema import Tags


def test_warp():
    np.random.seed(0)
    train, valid = generate_data("music-streaming", 100, (0.95, 0.05))
    train.schema = train.schema.remove_by_tag(Tags.TARGET)
    valid.schema = valid.schema.remove_by_tag(Tags.TARGET)

    model = LightFM(learning_rate=0.05, loss="warp", epochs=10)
    model.fit(train)

    model.predict(train)

    metrics = model.evaluate(valid, k=10)
    assert metrics["auc"] > 0.01
