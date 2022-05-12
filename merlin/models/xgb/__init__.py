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
        from merlin.models.data.movielens import get_movielens

        train, valid = get_movielens()

        # Train an XGBoost model
        from merlin.schema import Tags
        from merlin.models.xgb import XGBoost

        model = XGBoost(Tags.REGRESSION)
        model.fit(train)
    """

    def __init__(self, target_type: Tags, *args, **kwargs):
        self.target_type = target_type

    def fit(self, train: Dataset, params: Optional[dict] = None, **kwargs) -> xgb.Booster:
        """Trains the XGBoost Model

        Will use the columns tagged with the target_type passed to the
        constructor as the labels.  And all other non-list columns as
        input features.
        """
        target_columns = get_targets(train, self.target_type)

        # if the target is a regression, normalize values
        if self.target_type is Tags.REGRESSION:
            train = normalize_regession_targets(train)

        dtrain = dataset_to_dmatrix(train, target_columns)
        watchlist = [(dtrain, "train")]

        objective = get_objective(self.target_type)

        params = params or {}
        params.update(
            {
                "max_depth": 2,
                "eta": 1,
                "objective": objective,
                "eval_metric": "auc",
            }
        )

        self.bst = xgb.train(params, dtrain, evals=watchlist, **kwargs)

        return self.bst

    def evaluate(self, test_dataset: Dataset, k=10):
        pass


OBJECTIVES = {
    Tags.BINARY_CLASSIFICATION: "binary:logistic",
    Tags.REGRESSION: "reg:logistic",
}


def get_objective(target_type: Tags) -> str:
    # get the objective from the specified target type
    try:
        return OBJECTIVES[target_type]
    except KeyError:
        target_options_str = str(list(OBJECTIVES.keys()))
        raise ValueError(f"Target Objective not supported. Must be one of: {target_options_str}")


def get_targets(dataset: Dataset, target_type: Tags) -> List[str]:
    """Find target columns from dataset or specified target_column"""
    targets = dataset.schema.select_by_tag(Tags.TARGET).select_by_tag(target_type)

    if len(targets) >= 1:
        return targets.column_names
    else:
        raise ValueError(
            f"No target columns in the dataset schema with tags TARGET and {target_type.name}"
        )


def _to_numpy(series):
    """converts a pandas or cudf series to a numpy array"""
    if isinstance(series, pd.Series):
        return series.values
    else:
        return series.values_host


def dataset_to_dmatrix(dataset: Dataset, target_columns: List[str]) -> xgb.DMatrix:
    """Convert Merlin Dataset to XGBoost DMatrix"""
    df = dataset.to_ddf().compute(scheduler="synchronous")

    # Ignore list-like columns from schema
    list_column_names = [
        col_name
        for col_name, col_schema in dataset.schema.column_schemas.items()
        if col_schema.is_list
    ]

    X = _to_numpy(df.drop(target_columns + list_column_names, axis=1))
    y = _to_numpy(df[target_columns])

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
