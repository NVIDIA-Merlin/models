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
# from functools import lru_cache

from __future__ import absolute_import

from pathlib import Path

import distributed
import pytest

from merlin.core.utils import Distributed
from merlin.datasets.synthetic import generate_data
from merlin.io import Dataset

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def ecommerce_data() -> Dataset:
    return generate_data("e-commerce", num_rows=100)


@pytest.fixture
def music_streaming_data() -> Dataset:
    return generate_data("music-streaming", num_rows=100)


@pytest.fixture
def sequence_testing_data() -> Dataset:
    return generate_data("sequence-testing", num_rows=100)


@pytest.fixture
def social_data() -> Dataset:
    return generate_data("social", num_rows=100)


@pytest.fixture
def criteo_data() -> Dataset:
    return generate_data("criteo", num_rows=100)


@pytest.fixture
def testing_data() -> Dataset:
    data = generate_data("testing", num_rows=100)
    data.schema = data.schema.without(["session_id", "session_start", "day_idx"])

    return data


@pytest.fixture(scope="module")
def dask_client() -> distributed.Client:
    with Distributed(cluster_type="cpu") as dist:
        yield dist.client


try:
    import tensorflow as tf  # noqa

    from tests.unit.tf._conftest import *  # noqa
except ImportError:
    pass

try:
    import torchmetrics  # noqa

    from tests.unit.torch._conftest import *  # noqa
except ModuleNotFoundError:
    pass


def pytest_collection_modifyitems(items):
    for item in items:
        path = item.location[0]
        if "/integration/" in path:
            item.add_marker(pytest.mark.integration)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        if "/tf/" in path:
            item.add_marker(pytest.mark.tensorflow)
        if "/examples/" in path:
            item.add_marker(pytest.mark.example)
        if "/torch/" in path:
            item.add_marker(pytest.mark.torch)
        if "/implicit/" in path:
            item.add_marker(pytest.mark.implicit)
        if "/lightfm/" in path:
            item.add_marker(pytest.mark.lightfm)
        if "/xgb/" in path:
            item.add_marker(pytest.mark.xgboost)
        if "/datasets/" in path:
            item.add_marker(pytest.mark.datasets)
