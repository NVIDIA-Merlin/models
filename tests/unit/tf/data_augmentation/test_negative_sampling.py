import pandas as pd
import pytest
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.data_augmentation.negative_sampling import UniformNegativeSampling
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import ColumnSchema, Schema, Tags


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
        sampler = UniformNegativeSampling(schema, n_per_positive)

        input_df = pd.DataFrame(
            [
                {"user_id": 1, "item_id": 1, "item_feature": 2, "user_feature": 10, "label": 1},
                {"user_id": 2, "item_id": 3, "item_feature": 6, "user_feature": 5, "label": 1},
            ]
        )
        input_df = input_df[sorted(input_df.columns)]
        dataset = Dataset(input_df, schema=schema)
        batched_dataset = BatchedDataset(dataset, batch_size=10)
        batched_dataset = batched_dataset.map(sampler)

        feature_output, target_output = next(iter(batched_dataset))
        output_dict = {
            key: output_tensor.numpy().reshape(-1) for key, output_tensor in feature_output.items()
        }
        output_df = pd.DataFrame({**output_dict, "label": target_output.numpy().reshape(-1)})
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

    @pytest.mark.parametrize("to_dense", [True, False])
    def test_calling(self, music_streaming_data: Dataset, to_dense: bool, tf_random_seed: int):
        schema = music_streaming_data.schema
        batch_size, n_per_positive = 10, 5
        features = mm.sample_batch(
            music_streaming_data, batch_size=batch_size, include_targets=False, to_dense=to_dense
        )

        sampler = UniformNegativeSampling(schema, 5, seed=tf_random_seed)
        with_negatives = sampler(features)

        max_batch_size = batch_size + batch_size * n_per_positive
        assert all(
            f.shape[0] <= max_batch_size and f.shape[0] > batch_size
            for f in with_negatives.values()
        )

    def test_negatives_to_batch_in_model(self, music_streaming_data: Dataset, tf_random_seed: int):
        class Training(tf.keras.layers.Layer):
            def call(self, inputs, training=False):
                return training

        sampling = mm.Cond(
            Training(), UniformNegativeSampling(music_streaming_data.schema, 5, seed=tf_random_seed)
        )
        model = mm.Model(
            mm.InputBlock(music_streaming_data.schema),
            sampling,
            mm.MLPBlock([64]),
            mm.BinaryClassificationTask("click"),
        )

        batch_size = 10
        features, targets = mm.sample_batch(
            music_streaming_data, batch_size=batch_size, to_dense=True
        )

        with_negatives = model(features, targets=targets, training=True)
        assert with_negatives.predictions.shape[0] >= 50

        without_negatives = model(features)
        assert without_negatives.shape[0] == batch_size

    def test_model_with_dataloader(self, music_streaming_data: Dataset, tf_random_seed: int):
        add_negatives = UniformNegativeSampling(music_streaming_data.schema, 5, seed=tf_random_seed)

        batch_size, n_per_positive = 10, 5
        dataset = BatchedDataset(music_streaming_data, batch_size=batch_size)
        dataset = dataset.map(add_negatives)

        features, targets = next(iter(dataset))

        expected_batch_size = batch_size + batch_size * n_per_positive
        # We need to subtract the number of accidental hits
        expected_batch_size -= 4

        assert features["item_genres"].shape[0] == expected_batch_size
        assert all(f.shape[0] == expected_batch_size for f in features.values())
        assert all(f.shape[0] == expected_batch_size for f in targets.values())

        model = mm.Model(
            mm.InputBlock(music_streaming_data.schema),
            mm.MLPBlock([64]),
            mm.BinaryClassificationTask("click"),
        )
        assert model(features).shape[0] == expected_batch_size

        testing_utils.model_test(model, dataset)
