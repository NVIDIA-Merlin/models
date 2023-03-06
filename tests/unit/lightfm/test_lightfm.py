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
from pathlib import Path

import numpy as np
import pytest

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


def test_multiple_targets_raise_error():
    train, _ = generate_data("music-streaming", 100, (0.95, 0.05))

    model = LightFM()

    with pytest.raises(ValueError) as excinfo:
        model.fit(train)

    error_message = (
        "Found more than one column tagged Tags.TARGET in the dataset schema. "
        "Expected a single target column but found  ['click', 'play_percentage', 'like']"
    )
    assert error_message in str(excinfo.value)


def test_reload_no_target_column(tmpdir):
    np.random.seed(0)

    train, valid = generate_data("music-streaming", 100, (0.95, 0.05))
    train.schema = train.schema.remove_by_tag(Tags.TARGET)
    valid.schema = valid.schema.remove_by_tag(Tags.TARGET)

    model = LightFM(learning_rate=0.05, loss="warp", epochs=10)
    model.fit(train)

    model_dir = Path(tmpdir) / "lightfm_model"

    model.save(model_dir)
    reloaded = LightFM.load(model_dir)

    np.testing.assert_array_almost_equal(model.predict(valid), reloaded.predict(valid))

    assert reloaded.schema == model.schema
    assert reloaded.user_id_column == model.user_id_column
    assert reloaded.item_id_column == model.item_id_column
    assert reloaded.target_column is None


def test_reload_with_target_column(tmpdir):
    np.random.seed(0)

    train, valid = generate_data("music-streaming", 100, (0.95, 0.05))
    train.schema = train.schema.excluding_by_name(["play_percentage", "like"])
    valid.schema = valid.schema.excluding_by_name(["play_percentage", "like"])

    assert "click" in train.schema.column_names
    assert "click" in valid.schema.column_names

    model = LightFM(learning_rate=0.05, loss="warp", epochs=10)
    model.fit(train)

    model_dir = Path(tmpdir) / "lightfm_model"

    model.save(model_dir)
    reloaded = LightFM.load(model_dir)

    np.testing.assert_array_almost_equal(model.predict(valid), reloaded.predict(valid))

    assert reloaded.schema == model.schema
    assert reloaded.user_id_column == model.user_id_column
    assert reloaded.item_id_column == model.item_id_column
    assert reloaded.target_column == "click"
