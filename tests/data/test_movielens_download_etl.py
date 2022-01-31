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
import os

import cudf
import pytest

from merlin_standard_lib.utils.data_etl_utils import movielens_download_etl


@pytest.mark.parametrize("dataset_name", ["ml-100k", "ml-25m"])
def test_movielens_download_etl(tmp_path, dataset_name):
    movielens_download_etl(str(tmp_path), dataset_name)
    schema_file = os.path.join(tmp_path, dataset_name, "train/schema.pbtxt")
    print(schema_file)
    assert os.path.exists(schema_file)

    gdf = cudf.read_parquet(os.path.join(tmp_path, dataset_name, "train/part_0.parquet"))
    if dataset_name == "ml-100k":
        assert list(gdf.columns) == [
            "movieId",
            "userId",
            "genres",
            "imdb_URL",
            "TE_movieId_rating",
            "userId_count",
            "gender",
            "zip_code",
            "rating",
            "rating_binary",
            "age",
            "title",
        ]
        assert gdf.shape[0] == 90570
        assert gdf.isnull().any().sum() == 0
    elif dataset_name == "ml-25m":
        assert list(gdf.columns) == [
            "movieId",
            "userId",
            "genres",
            "TE_movieId_rating",
            "userId_count",
            "rating",
            "rating_binary",
            "title",
        ]
        assert gdf.shape[0] == 20000076
        assert gdf.isnull().any().sum() == 0
