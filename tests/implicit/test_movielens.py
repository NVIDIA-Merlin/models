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

pytest.importorskip("implicit")

from merlin.models.implicit import AlternatingLeastSquares  # noqa
from merlin.models.utils.data_etl_utils import get_movielens  # noqa


def test_implicit_movielens():
    # basic test: make sure we can train a model on movielens-25m and get a decent xval score on
    # it
    train, valid = get_movielens(variant="ml-25m")
    model = AlternatingLeastSquares(factors=128, iterations=15, regularization=0.01)
    model.fit(train)

    metrics = model.evaluate(valid)
    assert metrics["precision@10"] > 0.2
