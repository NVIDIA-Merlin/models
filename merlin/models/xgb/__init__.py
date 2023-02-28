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
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import distributed
import numpy as np
import xgboost as xgb

from merlin.core.utils import global_dask_client
from merlin.io import Dataset
from merlin.models.io import save_merlin_metadata
from merlin.models.utils.schema_utils import (
    schema_to_tensorflow_metadata_json,
    tensorflow_metadata_json_to_schema,
)
from merlin.schema import Schema, Tags


class XGBoost:
    """Create an XGBoost model from a merlin dataset.
    The class adapts an XGBoost model to work with the high level merlin-models API.

    Example usage::

        # get the movielens dataset
        from merlin.datasets.entertainment import get_movielens

        train, valid = get_movielens(variant="ml-1m")

        # Train an XGBoost model
        from merlin.core.utils import Distributed
        from merlin.models.xgb import XGBoost

        with Distributed():
            model = XGBoost(train.schema, objective="binary:logistic")
            model.fit(train)
            metrics = model.evaluate(valid)
    """

    def __init__(
        self,
        schema: Schema,
        *,
        target_columns: Optional[Union[str, list]] = None,
        qid_column: Optional[str] = None,
        objective: str = "reg:squarederror",
        booster: Optional[xgb.Booster] = None,
        **params,
    ):
        """
        Parameters
        ----------
        schema : merlin.schema.Schema
            The schema of the data that will be used to train and evaluate the model.
        target_columns : Optional[Union[list, str]]
            The target columns to use. If provided, will be used as the label(s).
            Otherwise the targets are automatically inferred from the objective and column tags.
        qid_column : Optional[str]
            For ranking objectives. The query ID column. If not provided will use
            the user ID (tagged with merlin.schema.Tags.USER_ID) column.
        objective : str
            The XGBoost objective to use. List of XGBoost objective functions:
            https://xgboost.readthedocs.io/en/stable/gpu/index.html#objective-functions
        **params
            The parameters to use for the XGBoost train method
        """
        self.params = {**params, "objective": objective}

        target_tag = get_target_tag(objective)
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        self.target_columns = target_columns or get_targets(schema, target_tag)
        self.feature_columns = get_features(schema, self.target_columns)

        if objective.startswith("rank") and qid_column is None:
            qid_column = schema.select_by_tag(Tags.USER_ID).column_names[0]
        self.qid_column = qid_column
        self.evals_result = {}
        self.booster = booster
        self.schema = schema

    @property
    def dask_client(self) -> Optional[distributed.Client]:
        return global_dask_client()

    def fit(
        self,
        train: Dataset,
        *,
        evals=None,
        use_quantile=True,
        **train_kwargs,
    ) -> xgb.Booster:
        """Trains the XGBoost Model.

        Will use the columns tagged with the target_type passed to the
        constructor as the labels.  And all other non-list columns as
        input features.

        Parameters
        ----------
        train : merlin.io.Dataset
            The training dataset to use to fit the model.
            We will use the column(s) tagged with merlin.schema.Tags.TARGET that match the
            objective as the label(s).
        evals : List[Tuple[Dataset, str]]
            List of tuples of datasets to watch
        use_quantile : bool
            This param is only relevant when using GPU.  (with
            tree_method="gpu_hist"). If set to False, will use a
            `DaskDMatrix`, instead of the default
            `DaskDeviceQuantileDMatrix`, which is preferred for GPU training.
        **train_kwargs
            Additional keyword arguments passed to the xgboost.train function

        Returns
        -------
        xgb.Booster

        Raises
        ------
        ValueError
           If objective is not supported. Or if the target columns cannot be found.
        """
        X, y, qid = dataset_to_xy(
            train,
            self.feature_columns,
            self.target_columns,
            self.qid_column,
        )

        dmatrix_cls = xgb.dask.DaskDMatrix
        if self.params.get("tree_method") == "gpu_hist" and use_quantile:
            # `DaskDeviceQuantileDMatrix` is a data type specialized
            # for the `gpu_hist` tree method that reduces memory overhead.
            # When training on GPU pipeline, it's preferred over `DaskDMatrix`.
            dmatrix_cls = xgb.dask.DaskDeviceQuantileDMatrix

        dtrain = dmatrix_cls(self.dask_client, X, label=y, qid=qid)
        watchlist = []

        if evals is None:
            evals = [(train, "train")]

        for _eval in evals:
            assert len(_eval) == 2
            dataset, name = _eval
            if dataset == train:
                watchlist.append((dtrain, name))
                continue
            assert isinstance(dataset, Dataset)
            X, y, qid = dataset_to_xy(
                dataset,
                self.feature_columns,
                self.target_columns,
                self.qid_column,
            )
            d_eval = dmatrix_cls(self.dask_client, X, label=y, qid=qid)
            watchlist.append((d_eval, name))

        train_res = xgb.dask.train(
            self.dask_client, self.params, dtrain, evals=watchlist, **train_kwargs
        )
        self.booster: xgb.Booster = train_res["booster"]
        self.evals_result = train_res["history"]

        return self.booster

    def evaluate(self, dataset: Dataset, **predict_kwargs) -> Dict[str, float]:
        """Evaluates the model on the dataset provided.

        Parameters
        ----------
        dataset : merlin.io.Dataset
            The dataset used to evaluate the model.
        **predict_kwargs
            The keyword parameters passed to the predict function

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics of the form {metric_name: value}.
        """
        if self.booster is None:
            raise ValueError("The fit method must be called before evaluate.")

        X, y, qid = dataset_to_xy(
            dataset, self.feature_columns, self.target_columns, self.qid_column
        )

        # convert to DMatrix
        # (eval doesn't have dask support currently)
        if qid is not None:
            qid = qid.compute()
        eval_data = xgb.DMatrix(X.compute(), label=y.compute(), qid=qid)

        metrics_str = self.booster.eval(eval_data)
        metrics = {}
        for metric in metrics_str.split("\t")[1:]:
            metric_name, metric_value = metric.split(":")
            metrics[metric_name[len("eval-") :]] = float(metric_value)

        return metrics

    def predict(self, dataset: Dataset, **predict_kwargs) -> np.ndarray:
        """Generate predictions from the dataset.

        Parameters
        ----------
        dataset : merlin.io.Dataset
            The dataset to use for predictions
        **predict_kwargs
            keyword arguments passed to the xgboost.core.Booster.predict method

        Returns
        -------
        numpy.ndarray
            The predicions data
        """
        if self.booster is None:
            raise ValueError("The fit method must be called before predict.")

        X, _, qid = dataset_to_xy(dataset, self.feature_columns, [], self.qid_column)
        data = xgb.dask.DaskDMatrix(self.dask_client, X, qid=qid)
        preds = xgb.dask.predict(self.dask_client, self.booster, data, **predict_kwargs).compute()

        return preds

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Save the model to a directory.

        Parameters
        ----------
        path : Union[str, os.PathLike]
            Directory where the model will be saved.
        """
        export_dir = Path(path)
        export_dir.mkdir(parents=True)
        self.booster.save_model(export_dir / "model.json")
        schema_to_tensorflow_metadata_json(self.schema, export_dir / "schema.json")

        save_merlin_metadata(
            export_dir,
            self.schema.select_by_name(self.feature_columns),
            self.schema.select_by_name(self.target_columns),
        )

        with open(export_dir / "params.json", "w") as f:
            json.dump(self.params, f, indent=4)
        with open(export_dir / "config.json", "w") as f:
            json.dump(
                dict(qid_column=self.qid_column, target_columns=self.target_columns), f, indent=4
            )

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "XGBoost":
        """Load the model from a directory where a model has been saved.

        Parameters
        ----------
        path : Union[str, os.PathLike]
            Path where a Merlin XGBoost model has been saved.

        Returns
        -------
        XGBoost model instance
        """
        load_dir = Path(path)
        booster = xgb.Booster()
        booster.load_model(load_dir / "model.json")
        schema = tensorflow_metadata_json_to_schema(load_dir / "schema.json")
        with open(load_dir / "params.json", "r") as f:
            params = json.load(f)
        with open(load_dir / "config.json", "r") as f:
            config = json.load(f)
        return cls(
            schema,
            target_columns=config.get("target_columns"),
            qid_column=config.get("qid_column"),
            booster=booster,
            **params,
        )


OBJECTIVES = {
    "binary:logistic": Tags.BINARY_CLASSIFICATION,
    "reg:logistic": Tags.REGRESSION,
    "reg:squarederror": Tags.REGRESSION,
    "rank:pairwise": Tags.TARGET,
    "rank:ndcg": Tags.TARGET,
    "rank:map": Tags.TARGET,
}


def get_target_tag(objective: str) -> Tags:
    """Get the target tag from the specified objective"""
    try:
        return OBJECTIVES[objective]
    except KeyError as exc:
        target_options_str = str(list(OBJECTIVES.keys()))
        raise ValueError(f"Objective not supported. Must be one of: {target_options_str}") from exc


def get_targets(schema: Schema, target_tag: Tags) -> List[str]:
    """Find target columns from dataset or specified target_column"""
    targets = schema.select_by_tag(Tags.TARGET).select_by_tag(target_tag)

    if len(targets) >= 1:
        return targets.column_names
    raise ValueError(
        f"No target columns in the dataset schema with tags TARGET and {target_tag.name}"
    )


def get_features(schema: Schema, target_columns: List[str]):
    """Find feature columns from schema. Returns all non-list column names from the schema
    that are not tagged as targets."""
    all_target_columns = schema.select_by_tag(Tags.TARGET).column_names + target_columns

    # Ignore list-like columns from schema
    list_column_names = [
        col_name for col_name, col_schema in schema.column_schemas.items() if col_schema.is_list
    ]

    if list_column_names:
        warnings.warn(f"Ignoring list columns as inputs to XGBoost model: {list_column_names}.")

    feature_columns = schema.excluding_by_name(
        set(list_column_names + all_target_columns)
    ).column_names
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found in schema.")

    return feature_columns


def dataset_to_xy(
    dataset: Dataset,
    feature_columns: List[str],
    target_columns: List[str],
    qid_column: Optional[str],
):
    """Convert Merlin Dataset to XGBoost DMatrix"""
    df = dataset.to_ddf()

    qid = None
    if qid_column:
        df = df.sort_values(qid_column)
        qid = df[qid_column]

    X = df[feature_columns]
    y = df[target_columns]

    return X, y, qid
