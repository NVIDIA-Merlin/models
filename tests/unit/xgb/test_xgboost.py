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
import pytest

from merlin.io import Dataset
from merlin.models.xgb import XGBoost


def test_without_dask_client(music_streaming_data: Dataset):
    with pytest.raises(ValueError) as exc_info:
        model = XGBoost(music_streaming_data.schema, objective="reg:logistic")
        model.fit(music_streaming_data)
    assert "No global client found" in str(exc_info.value)


@pytest.mark.usefixtures("dask_client")
class TestXGBoost:
    def test_unsupported_objective(self, music_streaming_data: Dataset):
        with pytest.raises(ValueError) as excinfo:
            model = XGBoost(music_streaming_data.schema, objective="reg:unknown")
            model.fit(music_streaming_data)
        assert "Objective not supported" in str(excinfo.value)

    def test_music_regression(self, music_streaming_data: Dataset):
        schema = music_streaming_data.schema
        model = XGBoost(schema, objective="reg:logistic")
        model.fit(music_streaming_data)
        model.predict(music_streaming_data)
        metrics = model.evaluate(music_streaming_data)

        assert "rmse" in metrics

    def test_ecommerce_click(self, ecommerce_data: Dataset):
        schema = ecommerce_data.schema
        model = XGBoost(
            schema, target_columns=["click"], objective="binary:logistic", eval_metric="auc"
        )
        model.fit(ecommerce_data)
        model.predict(ecommerce_data)
        metrics = model.evaluate(ecommerce_data)

        assert "auc" in metrics

    def test_social_click(self, social_data: Dataset):
        schema = social_data.schema
        model = XGBoost(
            schema, target_columns=["click"], objective="binary:logistic", eval_metric=["auc"]
        )
        model.fit(social_data)
        model.predict(social_data)
        metrics = model.evaluate(social_data)

        assert "auc" in metrics

    def test_logistic(self, criteo_data: Dataset):
        schema = criteo_data.schema
        model = XGBoost(schema, objective="binary:logistic", eval_metric=["auc"])
        model.fit(criteo_data)
        model.predict(criteo_data)
        metrics = model.evaluate(criteo_data)

        assert "auc" in metrics

    def test_ndcg(self, social_data: Dataset):
        schema = social_data.schema
        model = XGBoost(
            schema,
            target_columns="click",
            qid_column="user_id",
            objective="rank:ndcg",
            eval_metric=["auc", "ndcg", "map"],
        )
        model.fit(social_data)
        model.predict(social_data)
        metrics = model.evaluate(social_data)

        assert "map" in metrics

    def test_pairwise(self, social_data: Dataset):
        schema = social_data.schema
        model = XGBoost(
            schema,
            target_columns=["click"],
            qid_column="user_id",
            objective="rank:pairwise",
            eval_metric=["ndcg", "auc", "map"],
        )
        model.fit(social_data)
        model.predict(social_data)
        model.evaluate(social_data)
