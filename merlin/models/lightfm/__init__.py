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
import multiprocessing
import os
import pickle
from pathlib import Path
from typing import Optional, Union

import lightfm
import lightfm.evaluation

from merlin.io import Dataset
from merlin.models.io import save_merlin_metadata
from merlin.models.utils.dataset import (
    dataset_to_coo,
    get_item_id_column_name,
    get_target_column_name,
    get_user_id_column_name,
)
from merlin.models.utils.schema_utils import (
    schema_to_tensorflow_metadata_json,
    tensorflow_metadata_json_to_schema,
)
from merlin.schema import Schema


class LightFM:
    """
    This class adapts a model from lightfm to work with the high level merlin-models api

    Example usage::

        # Get the movielens dataset
        from merlin.datasets.entertainment import get_movielens

        train, valid = get_movielens()

        # Train a WARP model with lightfm using the merlin movielens dataset
        from merlin.models.lightfm import LightFM

        model = LightFM(learning_rate=0.05, loss="warp")
        model.fit(train)

        # evaluate the model given the validation set
        print(model.evaluate(valid))

        # save the model
        model.save("/path/to/dir")

        # reload the model
        model = LightFM.load("/path/to/dir")
    """

    def __init__(
        self,
        *args,
        epochs: int = 10,
        num_threads: int = 0,
        schema: Optional[Schema] = None,
        target_column: Optional[str] = None,
        lightfm_model: Optional[lightfm.LightFM] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        epochs: int
            Number of epochs to run.
        num_threads: int
            Number of parallel computation threads to use.
            Should not be higher than the number of physical cores.
        schema: merlin.schema.Schema
            The schema of the data that will be used to train and evaluate the model.
        target_column: Optional[str]
            The target column to use.
            If the schema contains multiple columns with Tags.TARGET, specify
            the column name to use for the target column.
        lightfm_model: Optional[lightfm.LightFM]
            If provided, an existing lightfm.LightFM instance will be loaded.
        """
        self.lightfm_model = lightfm_model or lightfm.LightFM(*args, **kwargs)
        self.epochs = epochs
        self.num_threads = num_threads or multiprocessing.cpu_count()
        self.schema = schema
        self.target_column = target_column
        self._select_features_and_column_from_schema()

    def _select_features_and_column_from_schema(self):
        if self.schema:
            self.user_id_column = get_user_id_column_name(self.schema)
            self.item_id_column = get_item_id_column_name(self.schema)
            self.target_column = self.target_column or get_target_column_name(self.schema)

    def fit(self, train: Dataset):
        """Trains the lightfm model

        Parameters
        ----------
        train : merlin.io.Dataset
            The training dataset to use to fit the model. We will use the the column tagged
            merlin.schema.Tags.ITEM_ID as the item , and merlin.schema.Tags.USER_ID as the userid.
            If there is a column tagged as Tags.TARGET we will also use that for the values,
            otherwise will be set to 1
        """
        if not self.schema:
            self.schema = train.schema
            self._select_features_and_column_from_schema()

        data = dataset_to_coo(train, target_column=self.target_column).tocsr()
        self.lightfm_model.fit(data, epochs=self.epochs, num_threads=self.num_threads)
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
            How many items to return per prediction
        """

        test = dataset_to_coo(test_dataset, target_column=self.target_column).tocsr()

        # lightfm needs the test set to have the same dimensionality as the train set
        test.resize(self.train_data.shape)

        precision = lightfm.evaluation.precision_at_k(
            self.lightfm_model, test, self.train_data, k=k, num_threads=self.num_threads
        ).mean()
        auc = lightfm.evaluation.auc_score(
            self.lightfm_model, test, self.train_data, num_threads=self.num_threads
        ).mean()
        return {f"precisions@{k}": precision, "auc": auc}

    def predict(self, dataset: Dataset, k=10):
        """Generate predictions from the dataset

        Parameters
        ----------
        test_dataset : merlin.io.Dataset
        k: int
            The number of recommendations to generate for each user
        """
        data = dataset_to_coo(dataset, target_column=self.target_column)
        return self.lightfm_model.predict(data.row, data.col)

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Saves the model to export_path using pickle, along with merlin
        model metadata.

        Parameters
        ----------
        path: Union[str, os.PathLike]
            Directory where the model will be saved.
        """
        export_dir = Path(path)
        export_dir.mkdir(parents=True)

        with open(export_dir / "lightfm_model.pkl", "wb") as f:
            pickle.dump(self.lightfm_model, f, protocol=pickle.HIGHEST_PROTOCOL)

        schema_to_tensorflow_metadata_json(self.schema, export_dir / "schema.json")
        save_merlin_metadata(
            export_dir,
            self.schema.select_by_name([self.user_id_column, self.item_id_column]),
            self.schema.select_by_name(self.target_column) if self.target_column else None,
        )

        with open(export_dir / "config.json", "w") as f:
            json.dump(
                dict(
                    epochs=self.epochs,
                    num_threads=self.num_threads,
                    target_column=self.target_column,
                ),
                f,
                indent=4,
            )

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "LightFM":
        """Load the model from a directory where a model has been saved.

        Parameters
        ----------
        path: Union[str, os.PathLike]
            Path where a Merlin LightFM model has been saved.

        Returns
        -------
        LightFM model instance.
        """
        load_dir = Path(path)
        schema = tensorflow_metadata_json_to_schema(load_dir / "schema.json")
        with open(load_dir / "lightfm_model.pkl", "rb") as f:
            lightfm_model = pickle.load(f)
        with open(load_dir / "config.json", "r") as f:
            config = json.load(f)
        return cls(
            epochs=config.get("epochs"),
            num_threads=config.get("num_threads"),
            schema=schema,
            lightfm_model=lightfm_model,
        )
