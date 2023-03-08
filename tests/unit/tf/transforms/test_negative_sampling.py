#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

import pandas as pd
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.core.prediction import Prediction
from merlin.models.tf.transforms.negative_sampling import InBatchNegatives
from merlin.models.tf.utils import testing_utils
from merlin.models.tf.utils.tf_utils import calculate_batch_size_from_inputs
from merlin.schema import ColumnSchema, Schema, Tags


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ExampleIsTraining(tf.keras.layers.Layer):
    def call(self, inputs, training=False):
        return training


@tf.keras.utils.register_keras_serializable(package="merlin.models")
class ExamplePredictionIdentity(tf.keras.layers.Layer):
    def call(self, inputs, targets=None):
        return Prediction(inputs, targets)

    def compute_output_shape(self, input_shape):
        return input_shape


class TestAddRandomNegativesToBatch:
    def test_dataloader(self):
        schema = Schema(
            [
                ColumnSchema(
                    "item_id", tags=[Tags.ITEM, Tags.ITEM_ID, Tags.CATEGORICAL], dtype="int64"
                ),
                ColumnSchema("item_feature", tags=[Tags.ITEM, Tags.CATEGORICAL]),
                ColumnSchema(
                    "user_id", tags=[Tags.USER, Tags.USER_ID, Tags.CATEGORICAL], dtype="int64"
                ),
                ColumnSchema("user_feature", tags=[Tags.USER, Tags.CATEGORICAL]),
                ColumnSchema("label", tags=[Tags.TARGET]),
            ]
        )
        n_per_positive = 5
        sampler = InBatchNegatives(schema, n_per_positive, prep_features=True)

        input_df = pd.DataFrame(
            [
                {"user_id": 1, "item_id": 1, "item_feature": 2, "user_feature": 10, "label": 1},
                {"user_id": 2, "item_id": 3, "item_feature": 6, "user_feature": 5, "label": 1},
            ]
        )
        input_df = input_df[sorted(input_df.columns)]
        dataset = Dataset(input_df, schema=schema)
        loader = mm.Loader(dataset, batch_size=10).map(sampler)
        outputs, targets = next(iter(loader))

        output_dict = {
            key: output_tensor.numpy().reshape(-1) for key, output_tensor in outputs.items()
        }
        output_df = pd.DataFrame({**output_dict, "label": targets.numpy().reshape(-1)})
        output_df = output_df[sorted(output_df.columns)]

        # first part of outputs frame should match inputs
        pd.testing.assert_frame_equal(input_df, output_df[: len(input_df)])

        # negatives added to batch should not overlap with positive user-items

        def user_item_pairs(df):
            return {tuple(x) for x in df[["user_id", "item_id"]].values.tolist()}

        input_user_item_pairs = user_item_pairs(input_df)
        negatives_df = output_df[len(input_df) :]
        negatives_user_item_pairs = user_item_pairs(negatives_df)

        assert input_user_item_pairs.intersection(negatives_user_item_pairs) == set()

    def assert_outputs_batch_size(self, assert_fn, outputs, targets=None):
        batch_size = calculate_batch_size_from_inputs(outputs)
        assert_fn(batch_size)

        if targets is not None:
            batch_size = calculate_batch_size_from_inputs(targets)
            assert_fn(batch_size)

    def test_calling_without_targets(self, music_streaming_data: Dataset, tf_random_seed: int):
        schema = music_streaming_data.schema
        batch_size, n_per_positive = 10, 5
        features = mm.sample_batch(
            music_streaming_data,
            batch_size=batch_size,
            include_targets=False,
            prepare_features=True,
        )

        sampler = InBatchNegatives(schema, n_per_positive, seed=tf_random_seed)

        outputs = sampler(features).outputs

        def assert_fn(output_batch_size):
            assert output_batch_size == batch_size

        self.assert_outputs_batch_size(assert_fn, outputs)

    def test_calling(self, music_streaming_data: Dataset, tf_random_seed: int):
        schema = music_streaming_data.schema
        batch_size, n_per_positive = 10, 5
        inputs, targets = mm.sample_batch(
            music_streaming_data,
            batch_size=batch_size,
            include_targets=True,
            prepare_features=True,
        )

        sampler = InBatchNegatives(schema, 5, seed=tf_random_seed)

        outputs, targets = sampler(inputs, targets=targets)

        max_batch_size = batch_size + batch_size * n_per_positive

        def assert_fn(output_batch_size):
            assert batch_size < output_batch_size <= max_batch_size

        self.assert_outputs_batch_size(
            assert_fn,
            outputs,
            targets,
        )

    def test_run_when_testing(self, music_streaming_data: Dataset, tf_random_seed: int):
        schema = music_streaming_data.schema
        batch_size, n_per_positive = 10, 5
        inputs, targets = mm.sample_batch(
            music_streaming_data,
            batch_size=batch_size,
            include_targets=True,
            prepare_features=True,
        )

        sampler = InBatchNegatives(
            schema, n_per_positive, seed=tf_random_seed, run_when_testing=False
        )

        with_negatives = sampler(inputs, targets=targets, testing=True)
        outputs = with_negatives.outputs
        targets = with_negatives.targets

        def assert_fn(output_batch_size):
            assert output_batch_size == batch_size

        self.assert_outputs_batch_size(
            assert_fn,
            outputs,
            targets,
        )

    # The sampling layer currnetly only works correctly as part of the model when run in eager mode
    @pytest.mark.parametrize("run_eagerly", [True])
    def test_in_model(self, run_eagerly, music_streaming_data: Dataset, tf_random_seed: int):
        dataset = music_streaming_data
        schema = dataset.schema

        sampling = InBatchNegatives(schema, 5, seed=tf_random_seed)

        model = mm.Model(
            mm.InputBlock(schema),
            sampling,
            mm.MLPBlock([64]),
            mm.BinaryClassificationTask("click"),
        )

        testing_utils.model_test(model, dataset, run_eagerly=run_eagerly, reload_model=True)

        batch_size = 10
        features, targets = mm.sample_batch(dataset, batch_size=batch_size, prepare_features=True)

        with_negatives = model(features, targets=targets, training=True)
        assert with_negatives.predictions.shape[0] >= 50

        without_negatives = model(features)
        assert without_negatives.shape[0] == batch_size

        preds = model.predict(features)
        assert preds.shape[0] == batch_size

    def test_model_with_dataloader(self, music_streaming_data: Dataset, tf_random_seed: int):
        dataset = music_streaming_data
        schema = dataset.schema

        add_negatives = InBatchNegatives(schema, 5, seed=tf_random_seed, prep_features=True)

        batch_size, n_per_positive = 10, 5
        loader = mm.Loader(dataset, batch_size=batch_size)

        features, targets = next(loader)
        features, targets = add_negatives(features, targets=targets)

        expected_batch_size = batch_size + batch_size * n_per_positive

        assert features["item_category"].shape[0] > batch_size
        assert features["item_category"].shape[0] <= expected_batch_size
        assert all(
            f.shape[0] > batch_size and f.shape[0] <= expected_batch_size for f in features.values()
        )
        assert all(
            f.shape[0] > batch_size and f.shape[0] <= expected_batch_size for f in targets.values()
        )

        model = mm.Model(
            mm.InputBlock(schema),
            mm.MLPBlock([64]),
            mm.BinaryClassificationTask("click"),
        )
        assert model(features).shape[0] > batch_size
        assert model(features).shape[0] <= expected_batch_size

        testing_utils.model_test(model, loader, fit_kwargs=dict(pre=add_negatives))
