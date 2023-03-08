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
import importlib
import json
import os
from pathlib import Path
from typing import Optional, Union

import implicit
from implicit.evaluation import ranking_metrics_at_k

from merlin.io import Dataset
from merlin.models.io import save_merlin_metadata
from merlin.models.utils.dataset import (
    _to_numpy,
    dataset_to_coo,
    get_item_id_column_name,
    get_user_id_column_name,
)
from merlin.models.utils.schema_utils import (
    schema_to_tensorflow_metadata_json,
    tensorflow_metadata_json_to_schema,
)
from merlin.schema import Schema


class ImplicitModelAdaptor:
    """
    This class adapts a model from implicit to work with the high level merlin-models api

    Example usage::

        # Get the movielens dataset
        from merlin.datasets.entertainment import get_movielens

        train, valid = get_movielens()

        # Train an ALS model with implicit using the merlin movielens dataset
        from merlin.models.implicit import AlternatingLeastSquares

        model = AlternatingLeastSquares(factors=128, iterations=15, regularization=0.01)
        model.fit(train)

        # evaluate the model given the validation set
        # prints out {'precision@10': 0.3895182175113377, 'map@10': 0.2609445966914911, ...} etc
        print(model.evaluate(valid))

        # save the model
        model.save("/path/to/dir")

        # reload the model
        model = AlternatingLeastSquares.load("/path/to/dir")
    """

    def __init__(self, implicit_model, schema: Optional[Schema] = None):
        self.implicit_model = implicit_model
        self.train_data = None
        self.schema = schema

    def fit(self, train: Dataset):
        """Trains the implicit model

        Parameters
        ----------
        train : merlin.io.Dataset
            The training dataset to use to fit the model. We will use the column tagged
            merlin.schema.Tags.ITEM_ID as the item , and merlin.schema.Tags.USER_ID as the userid.
            If there is a column tagged as Tags.TARGET we will also use that for the values,
            otherwise will be set to 1
        """
        if not self.schema:
            self.schema = train.schema
        data = dataset_to_coo(train).tocsr()
        self.implicit_model.fit(data)
        self.train_data = data

    def evaluate(self, test_dataset: Dataset, k=10):
        """Evaluates the model

        This function evaluates using a variety of ranking metrics, and returns
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
        test = dataset_to_coo(test_dataset).tocsr()
        ret = ranking_metrics_at_k(
            self.implicit_model,
            self.train_data,
            test,
            K=k,
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
        # Get the user-ids for the dataset,
        user_id_column = get_user_id_column_name(self.schema)
        userids = _to_numpy(
            dataset.to_ddf()[user_id_column].unique().compute(scheduler="synchronous")
        )

        return self.implicit_model.recommend(userids, None, filter_already_liked_items=False, N=k)

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Saves the model to directory 'path', along with merlin model metadata.

        Parameters
        ----------
        path: Union[str, os.PathLike]
            Directory where the model will be saved.
        """
        export_dir = Path(path)
        export_dir.mkdir(parents=True)

        self.implicit_model.save(export_dir / "implicit_model.npz")

        schema_to_tensorflow_metadata_json(self.schema, export_dir / "schema.json")

        user_id_column = get_user_id_column_name(self.schema)
        item_id_column = get_item_id_column_name(self.schema)
        save_merlin_metadata(
            export_dir,
            self.schema.select_by_name([user_id_column, item_id_column]),
            None,
        )
        with open(export_dir / "config.json", "w") as f:
            json.dump(
                dict(
                    implicit_model_module=self.implicit_model.__class__.__module__,
                    implicit_model_name=self.implicit_model.__class__.__name__,
                ),
                f,
                indent=4,
            )

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
    ):
        """Load the model from a directory where a model has been saved.

        Parameters
        ----------
        path: Union[str, os.PathLike]
            Path where a Merlin Implicit model has been saved.
        Returns
        -------
        Merlin Implicit model instance.
        """
        load_dir = Path(path)
        schema = tensorflow_metadata_json_to_schema(load_dir / "schema.json")

        with open(load_dir / "config.json", "r") as f:
            config = json.load(f)

        implicit_model_module = importlib.import_module(config.get("implicit_model_module"))
        implicit_model_name = config.get("implicit_model_name")
        implicit_model_cls = getattr(implicit_model_module, implicit_model_name)
        implicit_model = implicit_model_cls.load(load_dir / "implicit_model.npz")

        loaded_model = cls(schema=schema)
        loaded_model.implicit_model = implicit_model
        return loaded_model


class AlternatingLeastSquares(ImplicitModelAdaptor):
    def __init__(self, *args, schema: Optional[Schema] = None, **kwargs):
        """
        Parameters
        ----------
        schema: Optional[merlin.schema.Schema]
            The schema of the data that will be used to train and evaluate the model.
        """
        super().__init__(implicit.als.AlternatingLeastSquares(*args, **kwargs), schema=schema)


class BayesianPersonalizedRanking(ImplicitModelAdaptor):
    def __init__(self, *args, schema: Optional[Schema] = None, **kwargs):
        """
        Parameters
        ----------
        schema: Optional[merlin.schema.Schema]
            The schema of the data that will be used to train and evaluate the model.
        """
        super().__init__(implicit.bpr.BayesianPersonalizedRanking(*args, **kwargs), schema=schema)
