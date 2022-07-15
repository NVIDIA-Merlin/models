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

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.models.tf.utils.testing_utils import model_test
from merlin.schema import ColumnSchema, Tags


def test_embedding_features(tf_cat_features):
    dim = 15
    feature_config = {
        f: mm.FeatureConfig(mm.TableConfig(100, dim, name=f, initializer=None))
        for f in tf_cat_features.keys()
    }
    embeddings = mm.EmbeddingFeatures(feature_config)(tf_cat_features)

    assert list(embeddings.keys()) == list(feature_config.keys())
    assert all([emb.shape[-1] == dim for emb in embeddings.values()])


def test_embedding_features_tables():
    feature_config = {
        "cat_a": mm.FeatureConfig(mm.TableConfig(100, 32, name="cat_a", initializer=None)),
        "cat_b": mm.FeatureConfig(mm.TableConfig(64, 16, name="cat_b", initializer=None)),
    }
    embeddings = mm.EmbeddingFeatures(feature_config)

    assert embeddings.embedding_tables["cat_a"].input_dim == 100
    assert embeddings.embedding_tables["cat_a"].output_dim == 32

    assert embeddings.embedding_tables["cat_b"].input_dim == 64
    assert embeddings.embedding_tables["cat_b"].output_dim == 16


class TestEmbeddingTable:
    sample_column_schema = ColumnSchema(
        "item_id",
        dtype=np.int32,
        properties={"domain": {"min": 0, "max": 10, "name": "item_id"}},
        tags=[Tags.CATEGORICAL],
    )

    def test_raises_with_invalid_schema(self):
        column_schema = ColumnSchema(["item_id"])
        with pytest.raises(ValueError) as exc_info:
            mm.EmbeddingTable(16, column_schema)
        assert "needs to have a int-domain" in str(exc_info.value)

    @pytest.mark.parametrize(
        ["dim", "kwargs", "inputs", "expected_output_shape"],
        [
            (32, {}, tf.constant([1]), [1, 32]),
            (16, {}, tf.ragged.constant([[1, 2, 3], [4, 5]]), [2, None, 16]),
            (16, {"combiner": "mean"}, tf.ragged.constant([[1, 2, 3], [4, 5]]), [2, 16]),
            (16, {"combiner": "mean"}, tf.sparse.from_dense(tf.constant([[1, 2, 3]])), [1, 16]),
        ],
    )
    def test_layer(self, dim, kwargs, inputs, expected_output_shape):
        column_schema = self.sample_column_schema
        layer = mm.EmbeddingTable(dim, column_schema, **kwargs)

        output = layer(inputs)
        assert list(output.shape) == expected_output_shape

        if "combiner" in kwargs:
            assert isinstance(output, tf.Tensor)
        else:
            assert type(inputs) is type(output)

        layer_config = layer.get_config()
        copied_layer = mm.EmbeddingTable.from_config(layer_config)
        assert copied_layer.dim == layer.dim
        assert copied_layer.input_dim == layer.input_dim

        output = copied_layer(inputs)
        assert list(output.shape) == expected_output_shape

    def test_dense_with_combiner(self):
        dim = 16
        column_schema = self.sample_column_schema
        layer = mm.EmbeddingTable(dim, column_schema, combiner="mean")

        inputs = tf.constant([1])
        with pytest.raises(ValueError) as exc_info:
            layer(inputs)

        assert "Combiner only supported for RaggedTensor and SparseTensor." in str(exc_info.value)

    def test_sparse_without_combiner(self):
        dim = 16
        column_schema = self.sample_column_schema
        layer = mm.EmbeddingTable(dim, column_schema)

        inputs = tf.sparse.from_dense(tf.constant([[1, 2, 3]]))
        with pytest.raises(ValueError) as exc_info:
            layer(inputs)

        assert "EmbeddingTable supports only RaggedTensor and Tensor input types." in str(
            exc_info.value
        )

    def test_embedding_in_model(self, music_streaming_data: Dataset):
        dim = 16
        item_id_col_schema = music_streaming_data.schema.select_by_name("item_id").first
        embedding_layer = mm.EmbeddingTable(dim, item_id_col_schema)
        model = mm.Model(
            tf.keras.layers.Lambda(lambda inputs: inputs["item_id"]),
            embedding_layer,
            mm.BinaryClassificationTask("click"),
        )
        model_test(model, music_streaming_data)

    def test_non_trainable(self, music_streaming_data: Dataset):
        dim = 16
        item_id_col_schema = music_streaming_data.schema.select_by_name("item_id").first
        embedding_layer = mm.EmbeddingTable(dim, item_id_col_schema, trainable=False)
        inputs = tf.constant([1])

        model = mm.Model(
            tf.keras.layers.Lambda(lambda inputs: inputs["item_id"]),
            embedding_layer,
            mm.BinaryClassificationTask("click"),
        )
        model_test(model, music_streaming_data)

        output_before_fit = embedding_layer(inputs)
        embeddings_before_fit = embedding_layer.table.embeddings.numpy()

        model.fit(music_streaming_data, batch_size=50, epochs=1)

        output_after_fit = embedding_layer(inputs)
        embeddings_after_fit = embedding_layer.table.embeddings.numpy()

        np.testing.assert_array_almost_equal(output_before_fit, output_after_fit)
        np.testing.assert_array_almost_equal(embeddings_before_fit, embeddings_after_fit)

    @pytest.mark.parametrize("trainable", [True, False])
    def test_from_pretrained(self, trainable, music_streaming_data: Dataset):
        vocab_size = music_streaming_data.schema.column_schemas["item_id"].int_domain.max + 1
        embedding_dim = 32
        weights = np.random.rand(vocab_size, embedding_dim)
        pre_trained_weights_df = pd.DataFrame(weights)

        embedding_table = mm.EmbeddingTable.from_pretrained(
            pre_trained_weights_df, name="item_id", trainable=trainable
        )

        assert embedding_table.input_dim == vocab_size

        inputs = tf.constant([1])
        output = embedding_table(inputs)

        assert list(output.shape) == [1, embedding_dim]
        np.testing.assert_array_almost_equal(weights, embedding_table.table.embeddings)

        model = mm.Model(
            tf.keras.layers.Lambda(lambda inputs: inputs["item_id"]),
            embedding_table,
            mm.BinaryClassificationTask("click"),
        )
        model_test(model, music_streaming_data)

        if trainable:
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_almost_equal,
                weights,
                embedding_table.table.embeddings,
            )
        else:
            np.testing.assert_array_almost_equal(weights, embedding_table.table.embeddings)


def test_embedding_features_yoochoose(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = mm.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=mm.EmbeddingOptions(embedding_dim_default=512),
    )
    embeddings = emb_module(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert sorted(list(embeddings.keys())) == sorted(schema.column_names)
    assert all(emb.shape[-1] == 512 for emb in embeddings.values())
    max_value = list(schema.select_by_name("item_id"))[0].int_domain.max
    assert emb_module.embedding_tables["item_id"].embeddings.shape[0] == max_value + 1

    # These embeddings have not a specific initializer, so they should
    # have default truncated normal initialization
    default_truncated_normal_std = 0.05
    for emb_key in embeddings:
        assert embeddings[emb_key].numpy().mean() == pytest.approx(0.0, abs=0.02)
        assert embeddings[emb_key].numpy().std() == pytest.approx(
            default_truncated_normal_std, abs=0.04
        )


def test_serialization_embedding_features(testing_data: Dataset):
    inputs = mm.EmbeddingFeatures.from_schema(testing_data.schema)

    copy_layer = testing_utils.assert_serialization(inputs)

    assert list(inputs.feature_config.keys()) == list(copy_layer.feature_config.keys())

    from merlin.models.tf.inputs.embedding import serialize_table_config as serialize

    assert all(
        serialize(inputs.feature_config[key].table)
        == serialize(copy_layer.feature_config[key].table)
        for key in copy_layer.feature_config
    )


@testing_utils.mark_run_eagerly_modes
def test_embedding_features_yoochoose_model(music_streaming_data: Dataset, run_eagerly):
    schema = music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL)

    inputs = mm.EmbeddingFeatures.from_schema(schema, aggregation="concat")
    body = mm.SequentialBlock([inputs, mm.MLPBlock([64])])
    model = mm.Model(body, mm.BinaryClassificationTask("click"))

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


def test_embedding_features_yoochoose_custom_dims(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = mm.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=mm.EmbeddingOptions(
            embedding_dims={"item_id": 100}, embedding_dim_default=64
        ),
    )

    embeddings = emb_module(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert len(emb_module.losses) == 0, "There should be no regularization loss by default"

    assert emb_module.embedding_tables["item_id"].embeddings.shape[1] == 100
    assert emb_module.embedding_tables["categories"].embeddings.shape[1] == 64

    assert embeddings["item_id"].shape[1] == 100
    assert embeddings["categories"].shape[1] == 64


def test_embedding_features_l2_reg(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = mm.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=mm.EmbeddingOptions(
            embedding_dims={"item_id": 100}, embedding_dim_default=64, embeddings_l2_reg=0.1
        ),
    )

    _ = emb_module(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    l2_emb_losses = emb_module.losses

    assert len(l2_emb_losses) == len(
        schema
    ), "The number of reg losses should equal to the number of embeddings"

    for reg_loss in l2_emb_losses:
        assert reg_loss > 0.0


def test_embedding_features_yoochoose_infer_embedding_sizes(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = mm.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=mm.EmbeddingOptions(
            infer_embedding_sizes=True, infer_embedding_sizes_multiplier=3.0
        ),
    )

    embeddings = emb_module(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert (
        emb_module.embedding_tables["user_id"].embeddings.shape[1]
        == embeddings["user_id"].shape[1]
        == 20
    )
    assert (
        emb_module.embedding_tables["user_country"].embeddings.shape[1]
        == embeddings["user_country"].shape[1]
        == 9
    )
    assert (
        emb_module.embedding_tables["item_id"].embeddings.shape[1]
        == embeddings["item_id"].shape[1]
        == 46
    )
    assert (
        emb_module.embedding_tables["categories"].embeddings.shape[1]
        == embeddings["categories"].shape[1]
        == 13
    )


def test_embedding_features_yoochoose_infer_embedding_sizes_multiple_8(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = mm.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=mm.EmbeddingOptions(
            infer_embedding_sizes=True,
            infer_embedding_sizes_multiplier=3.0,
            infer_embeddings_ensure_dim_multiple_of_8=True,
        ),
    )

    embeddings = emb_module(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert (
        emb_module.embedding_tables["user_id"].embeddings.shape[1]
        == embeddings["user_id"].shape[1]
        == 24
    )
    assert (
        emb_module.embedding_tables["user_country"].embeddings.shape[1]
        == embeddings["user_country"].shape[1]
        == 16
    )
    assert (
        emb_module.embedding_tables["item_id"].embeddings.shape[1]
        == embeddings["item_id"].shape[1]
        == 48
    )
    assert (
        emb_module.embedding_tables["categories"].embeddings.shape[1]
        == embeddings["categories"].shape[1]
        == 16
    )


def test_embedding_features_yoochoose_partially_infer_embedding_sizes(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = mm.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=mm.EmbeddingOptions(
            embedding_dims={"user_id": 50, "user_country": 100},
            infer_embedding_sizes=True,
            infer_embedding_sizes_multiplier=3.0,
        ),
    )

    embeddings = emb_module(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert (
        emb_module.embedding_tables["user_id"].embeddings.shape[1]
        == embeddings["user_id"].shape[1]
        == 50
    )
    assert (
        emb_module.embedding_tables["user_country"].embeddings.shape[1]
        == embeddings["user_country"].shape[1]
        == 100
    )
    assert (
        emb_module.embedding_tables["item_id"].embeddings.shape[1]
        == embeddings["item_id"].shape[1]
        == 46
    )
    assert (
        emb_module.embedding_tables["categories"].embeddings.shape[1]
        == embeddings["categories"].shape[1]
        == 13
    )


def test_embedding_features_yoochoose_custom_initializers(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    random_max_abs_value = 0.3
    emb_module = mm.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=mm.EmbeddingOptions(
            embedding_dim_default=512,
            embeddings_initializers={
                "user_id": RandomUniform(minval=-random_max_abs_value, maxval=random_max_abs_value),
                "user_country": RandomUniform(
                    minval=-random_max_abs_value, maxval=random_max_abs_value
                ),
            },
        ),
    )

    embeddings = emb_module(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert embeddings["user_id"].numpy().min() == pytest.approx(-random_max_abs_value, abs=0.02)
    assert embeddings["user_id"].numpy().max() == pytest.approx(random_max_abs_value, abs=0.02)

    assert embeddings["user_country"].numpy().min() == pytest.approx(
        -random_max_abs_value, abs=0.02
    )
    assert embeddings["user_country"].numpy().max() == pytest.approx(random_max_abs_value, abs=0.02)

    # These embeddings have not a specific initializer, so they should
    # have default truncated normal initialization
    default_truncated_normal_std = 0.05
    assert embeddings["item_id"].numpy().mean() == pytest.approx(0.0, abs=0.02)
    assert embeddings["item_id"].numpy().std() == pytest.approx(
        default_truncated_normal_std, abs=0.04
    )

    assert embeddings["categories"].numpy().mean() == pytest.approx(0.0, abs=0.02)
    assert embeddings["categories"].numpy().std() == pytest.approx(
        default_truncated_normal_std, abs=0.04
    )


def test_embedding_features_yoochoose_pretrained_initializer(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    pretrained_emb_item_ids = np.random.random((51997, 64))
    pretrained_emb_categories = np.random.random((332, 64))

    emb_module = mm.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=mm.EmbeddingOptions(
            embeddings_initializers={
                "item_id": mm.TensorInitializer(pretrained_emb_item_ids),
                "categories": mm.TensorInitializer(pretrained_emb_categories),
            },
        ),
    )

    # Calling the first batch, so that embedding tables are build
    _ = emb_module(mm.sample_batch(testing_data, batch_size=10, include_targets=False))

    assert np.allclose(
        emb_module.embedding_tables["item_id"].embeddings.numpy(), pretrained_emb_item_ids
    )
    assert np.allclose(
        emb_module.embedding_tables["categories"].embeddings.numpy(), pretrained_emb_categories
    )


def test_embedding_features_exporting_and_loading_pretrained_initializer(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    emb_module = mm.EmbeddingFeatures.from_schema(schema)

    # Calling the first batch, so that embedding tables are build
    _ = emb_module(mm.sample_batch(testing_data, batch_size=10, include_targets=False))
    item_id_embeddings = emb_module.embedding_tables["item_id"].embeddings

    items_embeddings_dataset = emb_module.embedding_table_dataset(Tags.ITEM_ID, gpu=False)
    assert np.allclose(
        item_id_embeddings.numpy(), items_embeddings_dataset.to_ddf().compute().values
    )

    emb_init = mm.TensorInitializer.from_dataset(items_embeddings_dataset)
    assert np.allclose(item_id_embeddings.numpy(), emb_init(item_id_embeddings.shape).numpy())

    # Test GPU export if available
    try:
        import cudf  # noqa: F401

        items_embeddings_dataset = emb_module.embedding_table_dataset(Tags.ITEM_ID, gpu=True)
        assert np.allclose(
            item_id_embeddings.numpy(),
            items_embeddings_dataset.to_ddf().compute().to_pandas().values,
        )

        emb_init = mm.TensorInitializer.from_dataset(items_embeddings_dataset)
        assert np.allclose(item_id_embeddings.numpy(), emb_init(item_id_embeddings.shape).numpy())

    except ImportError:
        pass


def test_shared_embeddings(music_streaming_data: Dataset):
    inputs = mm.InputBlock(music_streaming_data.schema)

    embeddings = inputs.select_by_name(Tags.CATEGORICAL.value)

    assert embeddings.table_config("item_genres") == embeddings.table_config("user_genres")
