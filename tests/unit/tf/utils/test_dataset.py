#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
from merlin.core.dispatch import make_df
from merlin.io import Dataset
from merlin.models.utils.dataset import unique_by_tag, unique_rows_by_features
from merlin.schema import ColumnSchema, Schema, Tags


def test_unique_by_tag():
    df = make_df(
        {
            "user_id": [1, 1, 2, 3, 4, 3],
            "user_feat": [11, 11, 22, 33, 44, 33],
            "item_id": [45, 54, 23, 12, 45, 94],
        }
    )
    schema = Schema(
        [
            ColumnSchema("user_id", tags=[Tags.USER_ID]),
            ColumnSchema("user_feat", tags=[Tags.USER]),
            ColumnSchema("item_id", tags=[Tags.ITEM_ID]),
        ]
    )

    dataset = Dataset(df, schema=schema, npartitions=6)
    result_dataset = unique_by_tag(dataset, Tags.USER)

    result_df = result_dataset.compute()
    assert set(result_df.columns) == {"user_id", "user_feat"}
    assert result_df["user_id"].values.tolist() == [1, 2, 3, 4]
    assert result_df["user_feat"].values.tolist() == [11, 22, 33, 44]
    assert result_dataset.schema == schema.excluding_by_name(["item_id"])


def test_unique_rows_by_features():
    df = make_df(
        {
            "user_id": [1, 1, 2, 3, 4, 3],
            "user_feat": [11, 11, 22, 33, 44, 33],
            "item_id": [45, 54, 23, 12, 45, 94],
        }
    )
    schema = Schema(
        [
            ColumnSchema("user_id", tags=[Tags.USER_ID]),
            ColumnSchema("user_feat", tags=[Tags.USER]),
            ColumnSchema("item_id", tags=[Tags.ITEM_ID]),
        ]
    )

    dataset = Dataset(df, schema=schema, npartitions=6)
    result_dataset = unique_rows_by_features(dataset, Tags.USER)

    result_df = result_dataset.compute()
    assert set(result_df.columns) == {"user_id", "user_feat"}
    assert result_df["user_id"].values.tolist() == [1, 2, 3, 4]
    assert result_df["user_feat"].values.tolist() == [11, 22, 33, 44]
    assert result_dataset.schema == schema.excluding_by_name(["item_id"])
