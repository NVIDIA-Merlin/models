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

from merlin.datasets.synthetic import generate_data
from merlin.io import Dataset
from merlin.models.implicit import AlternatingLeastSquares, BayesianPersonalizedRanking
from merlin.schema import Tags


def test_alternating_least_squares(music_streaming_data: Dataset):
    np.random.seed(42)
    music_streaming_data.schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)

    model = AlternatingLeastSquares(factors=128, iterations=15, regularization=0.01)
    model.fit(music_streaming_data)
    metrics = model.evaluate(music_streaming_data)

    assert all(metric >= 0 for metric in metrics.values())

    model.predict(music_streaming_data)


def test_bayesian_personalized_ranking(music_streaming_data: Dataset):
    np.random.seed(42)
    music_streaming_data.schema = music_streaming_data.schema.remove_by_tag(Tags.TARGET)

    model = BayesianPersonalizedRanking(factors=128, iterations=15, regularization=0.01)
    model.fit(music_streaming_data)
    metrics = model.evaluate(music_streaming_data)

    assert all(metric >= 0 for metric in metrics.values())

    model.predict(music_streaming_data)


def test_reload_alternating_least_squares(music_streaming_data: Dataset, tmpdir):
    np.random.seed(42)
    train, valid = generate_data("music-streaming", 100, (0.95, 0.05))
    train.schema = train.schema.excluding_by_name(["play_percentage", "like"])
    valid.schema = valid.schema.excluding_by_name(["play_percentage", "like"])

    model = AlternatingLeastSquares(factors=128, iterations=15, regularization=0.01)
    model.fit(train)

    model_dir = Path(tmpdir) / "implicit_model"

    model.save(model_dir)
    reloaded = AlternatingLeastSquares.load(model_dir)

    np.testing.assert_array_almost_equal(model.predict(valid), reloaded.predict(valid))

    assert reloaded.schema == model.schema


def test_reload_bayesian_personalized_ranking(music_streaming_data: Dataset, tmpdir):
    np.random.seed(42)
    train, valid = generate_data("music-streaming", 100, (0.95, 0.05))
    train.schema = train.schema.excluding_by_name(["play_percentage", "like"])
    valid.schema = valid.schema.excluding_by_name(["play_percentage", "like"])

    model = BayesianPersonalizedRanking(factors=128, iterations=15, regularization=0.01)
    model.fit(train)

    model_dir = Path(tmpdir) / "implicit_model"

    model.save(model_dir)
    reloaded = BayesianPersonalizedRanking.load(model_dir)

    np.testing.assert_array_almost_equal(model.predict(valid), reloaded.predict(valid))

    assert reloaded.schema == model.schema
