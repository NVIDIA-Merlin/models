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
import multiprocessing

import lightfm
import lightfm.evaluation

from merlin.io import Dataset
from merlin.models.utils.dataset import dataset_to_coo


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
    """

    def __init__(self, *args, epochs=10, num_threads=0, **kwargs):
        self.lightfm_model = lightfm.LightFM(*args, **kwargs)
        self.epochs = epochs
        self.num_threads = num_threads or multiprocessing.cpu_count()

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
        data = dataset_to_coo(train).tocsr()
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

        test = dataset_to_coo(test_dataset).tocsr()

        # lightfm needs the test set to have the same dimensionality as the train set
        test.resize(self.train_data.shape)

        precision = lightfm.evaluation.precision_at_k(
            self.lightfm_model, test, self.train_data, k=k, num_threads=self.num_threads
        ).mean()
        auc = lightfm.evaluation.auc_score(
            self.lightfm_model, test, self.train_data, k=k, num_threads=self.num_threads
        ).mean()
        return {f"precisions@{k}": precision, f"auc@{k}": auc}

    def predict(self, dataset: Dataset, k=10):
        """Generate predictions from the dataset

        Parameters
        ----------
        test_dataset : merlin.io.Dataset
        k: int
            The number of recommendations to generate for each user
        """
        data = dataset_to_coo(dataset)
        return self.lightfm_model.predict(data.row, data.col)
