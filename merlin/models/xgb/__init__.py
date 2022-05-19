from typing import List, Optional

import nvtabular as nvt
import pandas as pd
import xgboost as xgb

from merlin.io import Dataset
from merlin.schema import Tags


class XGBoost:
    """
    The class adapts an XGBoost model to work with the high level merlin-models API.

    Example usage::

        # get the movielens dataset
        from merlin.datasets.entertainment import get_movielens

        train, valid = get_movielens()

        # Train an XGBoost model
        from merlin.schema import Tags
        from merlin.models.xgb import XGBoost

        model = XGBoost("binary:logistic")
        model.fit(train)

        model.evaluate(valid)
    """

    def __init__(self, objective: str, *args, **kwargs):
        """

        List of XGBoost objective functions:
        https://xgboost.readthedocs.io/en/stable/gpu/index.html#objective-functions
        """
        self.objective = objective

    def fit(self, train: Dataset, params: Optional[dict] = None, **kwargs) -> xgb.Booster:
        """Trains the XGBoost Model

        Will use the columns tagged with the target_type passed to the
        constructor as the labels.  And all other non-list columns as
        input features.
        """
        objective = self.objective
        target_tag = get_target_tag(objective)
        self.target_columns = get_targets(train, target_tag)

        # if the target is a regression, normalize values
        if objective == "reg:logistic":
            train = normalize_regession_targets(train)

        dtrain = dataset_to_dmatrix(train, self.target_columns)
        watchlist = [(dtrain, "train")]

        params = params or {}
        params.update(
            {
                "objective": objective,
            }
        )

        self.bst: xgb.Booster = xgb.train(params, dtrain, evals=watchlist, **kwargs)

        return self.bst

    def evaluate(self, test_dataset: Dataset):
        data: xgb.DMatrix = dataset_to_dmatrix(test_dataset, self.target_columns)
        preds = self.bst.predict(data)
        data.set_label(preds)
        metrics_str = self.bst.eval(data)
        metrics = {}
        for metric in metrics_str.split("\t")[1:]:
            metric_name, metric_value = metric.split(":")
            metrics[metric_name.removeprefix("eval-")] = float(metric_value)
        return metrics

    def predict(self, dataset: Dataset, **kwargs) -> np.ndarray:
        """Generate predictions from the dataset.

        Parameters
        ----------
        dataset : merlin.io.Dataset
            The dataset to use for predictions
        **kwargs
            keyword arguments passed to the xgboost.core.Booster.predict method

        Returns
        -------
        numpy.ndarray
            The predicions data
        """
        if self.bst is None:
            raise ValueError("The fit method must be called before predict.")

        data: xgb.DMatrix = dataset_to_dmatrix(dataset, self.target_columns)
        preds = self.bst.predict(data, **kwargs)
        return preds


OBJECTIVES = {
    "binary:logistic": Tags.BINARY_CLASSIFICATION,
    "reg:logistic": Tags.REGRESSION,
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


def dataset_to_dmatrix(dataset: Dataset, target_columns: List[str]) -> xgb.DMatrix:
    """Convert Merlin Dataset to XGBoost DMatrix"""
    df = dataset.to_ddf().compute()

    all_target_columns = dataset.schema.select_by_tag(Tags.TARGET).column_names

    # Ignore list-like columns from schema
    list_column_names = [
        col_name
        for col_name, col_schema in dataset.schema.column_schemas.items()
        if col_schema.is_list
    ]

    X = df.drop(all_target_columns + list_column_names, axis=1)
    y = df[target_columns]

    # Ensure columns are in a consistent order
    X = X[sorted(X.columns)]

    return xgb.DMatrix(X, label=y)


def normalize_regession_targets(dataset: Dataset) -> Dataset:
    """Normalize regression targets in a dataset to between 0 and 1"""
    regession_targets = dataset.schema.select_by_tag(Tags.TARGET).select_by_tag(Tags.REGRESSION)
    regression_features = regession_targets.column_names >> nvt.ops.NormalizeMinMax()
    workflow = nvt.Workflow(
        [c for c in dataset.schema.column_names if c not in regession_targets.column_names]
        + regression_features
    )
    workflow = workflow.fit(dataset)
    new_dataset = workflow.transform(dataset)
    return new_dataset
