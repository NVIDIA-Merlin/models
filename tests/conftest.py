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
from functools import lru_cache

import pytest

from merlin_models.data import synthetic

# LRU-cache for reading data from disk
synthetic._read_data = lru_cache(synthetic._read_data)


@pytest.fixture
def ecommerce_data() -> synthetic.SyntheticData:
    return synthetic.SyntheticData("e-commerce", num_rows=100)


@pytest.fixture
def music_streaming_data() -> synthetic.SyntheticData:
    return synthetic.SyntheticData("music_streaming", num_rows=100)


@pytest.fixture
def social_data() -> synthetic.SyntheticData:
    return synthetic.SyntheticData("social", num_rows=100)


@pytest.fixture
def testing_data() -> synthetic.SyntheticData:
    data = synthetic.SyntheticData("testing", num_rows=100)
    data._schema = data.schema.remove_by_name(["session_id", "session_start", "day_idx"])

    return data


try:
    import tensorflow as tf  # noqa

    from tests.tf.conftest import *  # noqa
except ImportError:
    pass

try:
    import torch  # noqa

    from tests.torch.conftest import *  # noqa
except ImportError:
    pass
