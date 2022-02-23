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

from merlin.models.data.synthetic import SyntheticData, generate_user_item_interactions


def test_generate_item_interactions_cpu(testing_data: SyntheticData):
    pd = pytest.importorskip("pandas")
    data = generate_user_item_interactions(
        testing_data.schema.without("event_timestamp"), num_interactions=500
    )

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


def test_generate_item_interactions_gpu(testing_data: SyntheticData):
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
