import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import xgboost as xgb

from merlin.io import Dataset
from merlin.schema import Tags


class XGBoost:
    """Create an XGBoost model.
    The class adapts an XGBoost model to work with the high level merlin-models API.

    Example usage::

        # get the movielens dataset
        from merlin.datasets.entertainment import get_movielens

        train, valid = get_movielens()

        # Train an XGBoost model
        from merlin.schema import Tags
        from merlin.models.xgb import XGBoost

        model = XGBoost(objective="binary:logistic")
        model.fit(train)

        model.evaluate(valid)
    """

    def __init__(self, *, objective="reg:squarederror", **params):
        """
        Parameters
        ----------
        objective : str
            The XGBoost objective to use. List of XGBoost objective functions:
            https://xgboost.readthedocs.io/en/stable/gpu/index.html#objective-functions
        **params
            The parameters to use for the XGBoost train method
        """
        self.params = {**params, "objective": objective}
        self.bst = None

    def fit(
        self,
        train: Dataset,
        *,
        target_columns: Optional[Union[str, list]] = None,
        qid_column: Optional[str] = None,
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
        target_columns : Optional[Union[list, str]]
            The target columns to use. If provided, will be used as the label(s).
            Otherwise the targets are automatically inferred from the objective and column tags.
        qid_column : Optional[str]
            For ranking objectives. The query ID column. If not provided will use
            the user ID (tagged with merlin.schema.Tags.USER_ID) column.
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
        objective = self.params["objective"]
        target_tag = get_target_tag(objective)
        self.target_columns = target_columns or get_targets(train, target_tag)

        # for ranking objectives, set the grouping
        if objective.startswith("rank") and qid_column is None:
            qid_column = train.schema.select_by_tag(Tags.USER_ID).column_names[0]
        self.qid_column = qid_column

        dtrain = dataset_to_dmatrix(train, self.target_columns, self.qid_column)
        watchlist = [(dtrain, "train")]

        self.bst = xgb.train(self.params, dtrain, evals=watchlist, **train_kwargs)

        return self.bst

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
        if self.bst is None:
            raise ValueError("The fit method must be called before evaluate.")

        data: xgb.DMatrix = dataset_to_dmatrix(dataset, self.target_columns, self.qid_column)
        preds = self.bst.predict(data, **predict_kwargs)
        data.set_label(preds)

        metrics_str = self.bst.eval(data)
        metrics = {}
        for metric in metrics_str.split("\t")[1:]:
            metric_name, metric_value = metric.split(":")
            metrics[metric_name.removeprefix("eval-")] = float(metric_value)

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
        if self.bst is None:
            raise ValueError("The fit method must be called before predict.")

        data: xgb.DMatrix = dataset_to_dmatrix(dataset, self.target_columns, self.qid_column)
        preds = self.bst.predict(data, **predict_kwargs)

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
    except KeyError:
        target_options_str = str(list(OBJECTIVES.keys()))
        raise ValueError(f"Objective not supported. Must be one of: {target_options_str}")


def get_targets(dataset: Dataset, target_tag: Tags) -> List[str]:
    """Find target columns from dataset or specified target_column"""
    targets = dataset.schema.select_by_tag(Tags.TARGET).select_by_tag(target_tag)

    if len(targets) >= 1:
        return targets.column_names
    else:
        raise ValueError(
            f"No target columns in the dataset schema with tags TARGET and {target_tag.name}"
        )


def dataset_to_dmatrix(
    dataset: Dataset, target_columns: Union[str, list], qid_column: Optional[str]
) -> xgb.DMatrix:
    """Convert Merlin Dataset to XGBoost DMatrix"""
    df = dataset.to_ddf().compute()

    qid = None
    if qid_column:
        df = df.sort_values(qid_column)
        qid = df[qid_column]

    all_target_columns = dataset.schema.select_by_tag(Tags.TARGET).column_names

    # Ignore list-like columns from schema
    list_column_names = [
        col_name
        for col_name, col_schema in dataset.schema.column_schemas.items()
        if col_schema.is_list
    ]

    if list_column_names:
        warnings.warn(f"Ignoring list columns as inputs to XGBoost model: {list_column_names}.")

    X = df.drop(all_target_columns + list_column_names, axis=1)
    y = df[target_columns]

    # Ensure columns are in a consistent order
    X = X[sorted(X.columns)]

    data = xgb.DMatrix(X, label=y, qid=qid)

    return data
