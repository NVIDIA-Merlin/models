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

import pytest

from merlin.models.data.synthetic import SyntheticData, _read_data

read_data = _read_data


@pytest.fixture
def ecommerce_data() -> SyntheticData:
    return SyntheticData("e-commerce", num_rows=100, read_data_fn=read_data)


@pytest.fixture
def music_streaming_data() -> SyntheticData:
    return SyntheticData("music_streaming", num_rows=100, read_data_fn=read_data)


@pytest.fixture
def sequence_testing_data() -> SyntheticData:
    return SyntheticData("sequence_testing", num_rows=100, read_data_fn=read_data)


@pytest.fixture
def social_data() -> SyntheticData:
    return SyntheticData("social", num_rows=100, read_data_fn=read_data)


@pytest.fixture
def testing_data() -> SyntheticData:
    data = SyntheticData("testing", num_rows=100, read_data_fn=read_data)
    data._schema = data.schema.without(["session_id", "session_start", "day_idx"])

    return data


try:
    import tensorflow as tf  # noqa

    from tests.tf._conftest import *  # noqa
except ImportError:
    pass

try:
    import torchmetrics  # noqa

    from tests.torch._conftest import *  # noqa
except ModuleNotFoundError:
    pass
