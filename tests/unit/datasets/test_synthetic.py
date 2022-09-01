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
import pytest

from merlin.datasets.synthetic import generate_data, generate_user_item_interactions
from merlin.io import Dataset
from merlin.models.utils.schema_utils import filter_dict_by_schema
from merlin.schema import ColumnSchema, Schema, Tags


def test_synthetic_sequence_testing_data():
    dataset = generate_data("testing", 100)

    assert isinstance(dataset, Dataset)
    assert dataset.num_rows == 100
    assert len(dataset.schema) == 11


def test_train_valid_testing_data():
    train, valid = generate_data("testing", 1000, set_sizes=(0.8, 0.2))

    assert isinstance(train, Dataset)
    assert isinstance(valid, Dataset)
    assert train.num_rows == 800
    assert valid.num_rows == 200


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


def test_generate_user_item_interactions_dtypes():
    schema = Schema(
        [
            ColumnSchema(
                "item_id",
                dtype=np.int32,
                tags=[Tags.ITEM_ID],
                properties={"domain": {"min": 0, "max": 10}},
            ),
            ColumnSchema("f32", dtype=np.float32),
            ColumnSchema("f64", dtype=np.float64),
            ColumnSchema("i16", dtype=np.int16),
            ColumnSchema("i32", dtype=np.int32),
            ColumnSchema("i64", dtype=np.int64),
            ColumnSchema("bool", dtype=np.dtype("bool")),
        ]
    )
    data = generate_user_item_interactions(schema, num_interactions=5)
    assert data["f32"].dtype == np.float32
    assert data["f64"].dtype == np.float64
    assert data["i16"].dtype == np.int16
    assert data["i32"].dtype == np.int32
    assert data["i64"].dtype == np.int64
    assert data["bool"].dtype == np.dtype("bool")


def test_generate_item_interactions_cpu(testing_data: Dataset):
    pd = pytest.importorskip("pandas")
    schema = testing_data.schema.without("event_timestamp")
    data = generate_user_item_interactions(schema, num_interactions=500)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 500
    assert list(data.columns) == [
        "user_id",
        "user_country",
        "user_age",
        "item_id",
        "item_age_days_norm",
        "event_hour_sin",
        "event_hour_cos",
        "event_weekday_sin",
        "event_weekday_cos",
        "categories",
    ]
    expected_dtypes = {
        "user_id": "int64",
        "user_country": "int64",
        "user_age": "float64",
        "item_id": "int64",
        "item_age_days_norm": "float64",
        "event_hour_sin": "float64",
        "event_hour_cos": "float64",
        "event_weekday_sin": "float64",
        "event_weekday_cos": "float64",
        "categories": "int64",
    }

    assert all(
        val == expected_dtypes[key] for key, val in dict(data.dtypes).items() if key != "categories"
    )


def test_generate_item_interactions_gpu(testing_data: Dataset):
    cudf = pytest.importorskip("cudf")
    data = generate_user_item_interactions(testing_data.schema, num_interactions=500, device="cuda")

    assert isinstance(data, cudf.DataFrame)
    assert len(data) == 500
    assert list(data.columns) == [
        "user_id",
        "user_country",
        "user_age",
        "item_id",
        "item_age_days_norm",
        "event_hour_sin",
        "event_hour_cos",
        "event_weekday_sin",
        "event_weekday_cos",
        "categories",
        "event_timestamp",
    ]
    expected_dtypes = {
        "user_id": "int64",
        "user_country": "int64",
        "user_age": "float64",
        "item_id": "int64",
        "item_age_days_norm": "float64",
        "event_hour_sin": "float64",
        "event_hour_cos": "float64",
        "event_weekday_sin": "float64",
        "event_weekday_cos": "float64",
        "categories": "int64",
        "event_timestamp": "int64",
    }

    assert all(
        val == expected_dtypes[key] for key, val in dict(data.dtypes).items() if key != "categories"
    )
