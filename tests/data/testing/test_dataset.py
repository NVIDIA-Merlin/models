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

from merlin.models.data.synthetic import SyntheticData
from merlin.models.utils.schema import filter_dict_by_schema
from merlin.schema import Tags
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata


def test_tabular_sequence_testing_data():
    tabular_testing_data = SyntheticData("testing")
    assert isinstance(tabular_testing_data, SyntheticData)

    assert tabular_testing_data.schema_path.endswith("merlin/models/data/testing/schema.json")
    assert len(tabular_testing_data.schema) == 11


def test_tabular_music_data():
    tabular_music_data = SyntheticData("music_streaming")
    assert isinstance(tabular_music_data, SyntheticData)
    data = tabular_music_data.generate_interactions(100)

    assert data["position"].shape == (100,)
    targets = tabular_music_data.schema.select_by_tag(Tags.TARGET)
    assert len(targets) == 3
    for val in targets:
        assert data[val.name].shape == (100,)


def test_tf_tensors_generation_cpu():
    tf = pytest.importorskip("tensorflow")
    tabular_testing_data = SyntheticData("testing")
    schema = tabular_testing_data.schema
    data = tabular_testing_data.generate_interactions(
        num_rows=100, min_session_length=5, max_session_length=50, save=False
    ).to_dict("list")

    tensors = {key: tf.convert_to_tensor(value) for key, value in data.items()}
    assert tensors["user_id"].shape == (100,)
    assert tensors["user_age"].dtype == tf.float32
    for name, val in filter_dict_by_schema(tensors, schema.select_by_tag(Tags.LIST)).items():
        assert val.shape == (100, 50)

    for val in filter_dict_by_schema(tensors, schema.select_by_tag(Tags.CATEGORICAL)).values():
        assert val.dtype == tf.int32
        assert tf.reduce_max(val) < 52000


def test_torch_tensors_generation_cpu():
    torch = pytest.importorskip("torch")
    tabular_testing_data = SyntheticData("testing")
    schema = tabular_testing_data.schema
    data = tabular_testing_data.generate_interactions(
        num_rows=100, min_session_length=5, max_session_length=50, save=False
    ).to_dict("list")
    tensors = {key: torch.tensor(value) for key, value in data.items()}

    assert tensors["user_id"].shape == (100,)
    assert tensors["user_age"].dtype == torch.float32
    for val in filter_dict_by_schema(tensors, schema.select_by_tag(Tags.LIST)).values():
        assert val.shape == (100, 50)

    for val in filter_dict_by_schema(tensors, schema.select_by_tag(Tags.CATEGORICAL)).values():
        assert val.dtype == torch.int64
        assert val.max() < 52000


def test_synthetic_read_proto_text(tmpdir):
    schema = SyntheticData("music_streaming").schema
    TensorflowMetadata.from_merlin_schema(schema).to_proto_text_file(tmpdir, "schema.pbtxt")
    reloaded = SyntheticData.read_schema(tmpdir / "schema.pbtxt")
    assert schema == reloaded
