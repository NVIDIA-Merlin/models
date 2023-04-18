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
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import sklearn.datasets
import xgboost

from merlin.core.compat import HAS_GPU
from merlin.datasets.synthetic import generate_data
from merlin.io import Dataset
from merlin.models.xgb import XGBoost, dataset_to_xy


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


@pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
@pytest.mark.parametrize(
    ["fit_kwargs", "expected_dtrain_cls"],
    [
        ({}, xgboost.dask.DaskDeviceQuantileDMatrix),
        ({"use_quantile": False}, xgboost.dask.DaskDMatrix),
    ],
)
@patch("xgboost.dask.train", side_effect=xgboost.dask.train)
def test_gpu_hist_dmatrix(
    mock_train, fit_kwargs, expected_dtrain_cls, dask_client, music_streaming_data: Dataset
):
    schema = music_streaming_data.schema
    model = XGBoost(schema, objective="reg:logistic", tree_method="gpu_hist")
    model.fit(music_streaming_data, **fit_kwargs)
    model.predict(music_streaming_data)
    metrics = model.evaluate(music_streaming_data)
    assert "rmse" in metrics

    assert mock_train.called
    assert mock_train.call_count == 1

    train_call = mock_train.call_args_list[0]
    client, params, dtrain = train_call.args
    assert dask_client == client
    assert params["tree_method"] == "gpu_hist"
    assert params["objective"] == "reg:logistic"
    assert isinstance(dtrain, expected_dtrain_cls)


@pytest.mark.usefixtures("dask_client")
class TestSchema:
    def test_fit_with_sub_schema(self, music_streaming_data: Dataset):
        schema = music_streaming_data.schema
        sub_schema = schema.select_by_name(["session_id", "country", "play_percentage"])
        model = XGBoost(sub_schema, objective="reg:logistic")
        model.fit(music_streaming_data)
        assert model.booster.num_features() == 2

    def test_no_features(self, music_streaming_data: Dataset):
        schema = music_streaming_data.schema
        sub_schema = schema.select_by_name(["unknown_feature", "play_percentage"])
        with pytest.raises(ValueError) as excinfo:
            XGBoost(sub_schema, objective="reg:logistic")
        assert "No feature columns found" in str(excinfo.value)

    def test_fit_with_missing_features(self, music_streaming_data: Dataset):
        schema = music_streaming_data.schema
        sub_schema = schema.select_by_name(["session_id", "play_percentage"])
        model = XGBoost(sub_schema, objective="reg:logistic")
        df = music_streaming_data.to_ddf().compute()
        new_dataset = Dataset(df[["click", "play_percentage"]])
        with pytest.raises(KeyError) as excinfo:
            model.fit(new_dataset)
        assert "session_id" in str(excinfo)


class TestEvals:
    def test_multiple(self, dask_client):
        train, valid_a, valid_b = generate_data(
            "music-streaming", num_rows=100, set_sizes=(0.6, 0.2, 0.2)
        )
        model = XGBoost(train.schema, objective="reg:logistic")
        model.fit(train, evals=[(valid_a, "a"), (valid_b, "b")])
        assert set(model.evals_result.keys()) == {"a", "b"}

    def test_default(self, dask_client):
        train = generate_data("music-streaming", num_rows=100)
        model = XGBoost(train.schema, objective="reg:logistic")
        model.fit(train)
        assert set(model.evals_result.keys()) == {"train"}

    def test_train_and_valid(self, dask_client):
        train, valid = generate_data("music-streaming", num_rows=100, set_sizes=(0.5, 0.5))
        model = XGBoost(train.schema, objective="reg:logistic")
        model.fit(train, evals=[(valid, "valid"), (train, "train")])
        assert set(model.evals_result.keys()) == {"valid", "train"}

    def test_invalid_data(self, dask_client):
        train, _ = generate_data("music-streaming", num_rows=100, set_sizes=(0.5, 0.5))
        model = XGBoost(train.schema, objective="reg:logistic")
        with pytest.raises(AssertionError):
            model.fit(train, evals=[([], "valid")])


def test_dataset_to_xy_does_not_modify_column_order():
    df = pd.DataFrame(data={"z": [0], "target": [-1], "a": [1], "Z": [2]})
    feature_columns = ["z", "Z", "a"]
    X, y, _ = dataset_to_xy(
        dataset=Dataset(df),
        feature_columns=feature_columns,
        target_columns="target",
        qid_column=None,
    )
    assert X.columns.tolist() == feature_columns


def test_predict_without_target(dask_client):
    rows = 200
    num_features = 16
    X, y = sklearn.datasets.make_regression(
        n_samples=rows,
        n_features=num_features,
        n_informative=num_features // 3,
        random_state=0,
    )

    feature_names = [str(i) for i in range(num_features)]
    df = pd.DataFrame(
        np.hstack((X, y.reshape(-1, 1))), columns=feature_names + ["target"], dtype=np.float32
    )
    train_ds = Dataset(df.iloc[:160])
    valid_ds = Dataset(df.iloc[40:])
    test_ds = Dataset(df.iloc[40:].drop(columns="target"))

    model = XGBoost(schema=train_ds.schema, target_columns="target")
    model.fit(train_ds, evals=[(valid_ds, "validation_set")])
    model.predict(test_ds)


def test_reload(dask_client, tmpdir):
    train, valid = generate_data("social", num_rows=40, set_sizes=(0.5, 0.5))

    schema = train.schema
    xgb_booster_params = {
        "objective": "rank:pairwise",
    }

    xgb_train_params = {
        "num_boost_round": 1,
        "verbose_eval": 1,
        "early_stopping_rounds": 1,
    }

    model = XGBoost(schema, target_columns=["click"], qid_column="user_id", **xgb_booster_params)
    model.fit(
        train,
        evals=[
            (valid, "validation_set"),
        ],
        **xgb_train_params
    )
    _ = model.evaluate(valid)

    model_dir = Path(tmpdir) / "xgb_model"

    model.save(model_dir)
    reloaded = XGBoost.load(model_dir)

    np.testing.assert_array_almost_equal(model.predict(valid), reloaded.predict(valid))

    assert reloaded.schema == model.schema
    assert reloaded.target_columns == model.target_columns
    assert reloaded.feature_columns == model.feature_columns
    assert reloaded.qid_column == model.qid_column
