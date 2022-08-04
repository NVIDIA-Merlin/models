import warnings
from typing import Dict, List, Optional, Union

import distributed
import numpy as np
import xgboost as xgb

from merlin.core.utils import global_dask_client
from merlin.io import Dataset
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
