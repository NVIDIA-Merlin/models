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
import implicit
import numpy as np
import pandas as pd
from implicit.evaluation import ranking_metrics_at_k
from scipy.sparse import coo_matrix

from merlin.io import Dataset
from merlin.schema import Tags


class ImplicitModelAdaptor:
    """
    This class adapts a model from implicit to work with the high level merlin-models api

    Example usage::

        # Get the movielens dataset
        from merlin.models.utils.data_etl_utils import get_movielens

        train, valid = get_movielens()

        # Train a ALS model with implicit using the merlin movielens dataset
        from merlin.models.implicit import AlternatingLeastSquares

        model = AlternatingLeastSquares(factors=128, iterations=15, regularization=0.01)
        model.fit(train)

        # evaluate the model given the validation set
        # prints out {'precision@10': 0.3895182175113377, 'map@10': 0.2609445966914911, ...} etc
        print(model.evaluate(valid))
    """

    def __init__(self, implicit_model):
        self.implicit_model = implicit_model
        self.train_data = None

    def fit(self, train: Dataset):
        """Trains the implicit model

        Parameters
        ----------
        train : merlin.io.Dataset
            The training dataset to use to fit the model. We will use the the column tagged
            merlin.schema.Tags.ITEM_ID as the item , and merlin.schema.Tags.USER_ID as the userid.
            If there is a column tagged as Tags.TARGET we will also use that for the values,
            otherwise will be set to 1
        """
        data = _dataset_to_coo(train).tocsr()
        self.implicit_model.fit(data)
        self.train_data = data

    def evaluate(self, test_dataset: Dataset, k=10):
        """Evaluates the model

        This function evalutes using a variety of ranking metrics, and returns
        a dictionary of {metric_name: value}.

        Parameters
        ----------
        test_dataset : merlin.io.Dataset
            The validation dataset to evaluate
        k : int
            How many items to return per prediction. By default this method will
            return metrics like 'map@10' , but by increasing k you can generate
            different versions
        """
        test = _dataset_to_coo(test_dataset).tocsr()
        ret = ranking_metrics_at_k(
            self.implicit_model, self.train_data, test, K=k, filter_already_liked_items=True
        )
        return {metric + f"@{k}": value for metric, value in ret.items()}

    def predict(self, dataset: Dataset, k=10):
        """Generate predictions from the dataset
        Parameters
        ----------
        test_dataset : merlin.io.Dataset
            The dataset to use for generating predictions. Each userid (as denoted by Tags.USER_ID
            will get recommendations generated for them
        k: int
            The number of recommendations to generate for each user
        """
        # Get the userids for the dataset,
        user_id_column = dataset.schema.select_by_tag(Tags.USER_ID).first.name
        userids = _to_numpy(
            dataset.to_ddf()[user_id_column].unique().compute(scheduler="synchronous")
        )

        return self.implicit_model.recommend(userids, None, filter_already_liked_items=False, N=k)


class AlternatingLeastSquares(ImplicitModelAdaptor):
    def __init__(self, *args, **kwargs):
        super().__init__(implicit.als.AlternatingLeastSquares(*args, **kwargs))


class BayesianPersonalizedRanking(ImplicitModelAdaptor):
    def __init__(self, *args, **kwargs):
        super().__init__(implicit.bpr.BayesianPersonalizedRanking(*args, **kwargs))


def _dataset_to_coo(dataset: Dataset):
    """Converts a merlin.io.Dataset object to a scipy coo matrix"""
    user_id_column = dataset.schema.select_by_tag(Tags.USER_ID).first.name
    item_id_column = dataset.schema.select_by_tag(Tags.ITEM_ID).first.name

    columns = [user_id_column, item_id_column]
    target_column = None
    target = dataset.schema.select_by_tag(Tags.TARGET)

    if len(target) > 1:
        raise ValueError(
            "Found more than one column tagged Tags.TARGET in the dataset schema."
            f" Expected a single target column but found  {target.column_names}"
        )

    elif len(target) == 1:
        target_column = target.first.name
        columns.append(target_column)

    df = dataset.to_ddf()[columns].compute(scheduler="synchronous")

    userids = _to_numpy(df[user_id_column])
    itemids = _to_numpy(df[item_id_column])
    targets = _to_numpy(df[target_column]) if target_column else np.ones(len(userids))
    return coo_matrix((targets.astype("float32"), (userids, itemids)))


def _to_numpy(series):
    """converts a pandas or cudf series to a numpy array"""
    if isinstance(series, pd.Series):
        return series.values
    else:
        return series.values_host
