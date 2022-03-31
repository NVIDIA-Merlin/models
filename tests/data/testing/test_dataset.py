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

from merlin.io import Dataset
from merlin.models.data.synthetic import generate_data
from merlin.models.utils.schema_utils import filter_dict_by_schema
from merlin.schema import Tags


def test_tabular_sequence_testing_data():
    tabular_testing_data = generate_data("testing", 100)
    assert isinstance(tabular_testing_data, Dataset)

    assert len(tabular_testing_data.schema) == 11


def test_tabular_music_data():
    tabular_music_data = generate_data("music-streaming", 100)
    assert isinstance(tabular_music_data, Dataset)


def test_tf_tensors_generation_cpu():
    tf = pytest.importorskip("tensorflow")
    data = generate_data("testing", num_rows=100, min_session_length=5, max_session_length=50)
    schema = data.schema

    from merlin.models.tf import sample_batch

    tensors = sample_batch(data, batch_size=100, include_targets=False)
    assert tensors["user_id"].shape == (100, 1)
    assert tensors["user_age"].dtype == tf.float64
    for name, val in filter_dict_by_schema(tensors, schema.select_by_tag(Tags.LIST)).items():
        assert len(val) == 2


# def test_torch_tensors_generation_cpu():
#     torch = pytest.importorskip("torch")
#     tabular_testing_data = SyntheticData("testing")
#     schema = tabular_testing_data.schema
#     data = tabular_testing_data.generate_interactions(
#         num_rows=100, min_session_length=5, max_session_length=50, save=False
#     ).to_dict("list")
#     tensors = {key: torch.tensor(value) for key, value in data.items()}
#
#     assert tensors["user_id"].shape == (100,)
#     assert tensors["user_age"].dtype == torch.float32
#     for val in filter_dict_by_schema(tensors, schema.select_by_tag(Tags.LIST)).values():
#         assert val.shape == (100, 50)
#
#     for val in filter_dict_by_schema(tensors, schema.select_by_tag(Tags.CATEGORICAL)).values():
#         assert val.dtype == torch.int64
#         assert val.max() < 52000
