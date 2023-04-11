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
from merlin.core.compat import cudf
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.models.tf.utils.testing_utils import assert_output_shape, model_test
from merlin.schema import ColumnSchema, Schema, Tags


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
        column_schema = ColumnSchema("item_id")
        with pytest.raises(ValueError) as exc_info:
            mm.EmbeddingTable(16, column_schema)
        assert "needs to have an int-domain" in str(exc_info.value)

    @pytest.mark.parametrize(
        ["dim", "kwargs", "inputs", "expected_output_shape"],
        [
            (32, {}, tf.constant([[1]]), [1, 32]),
            (16, {}, tf.ragged.constant([[[1], [2], [3]], [[4], [5]]]), [2, None, 16]),
            (
                16,
                {"sequence_combiner": "mean"},
                tf.ragged.constant([[[1], [2], [3]], [[4], [5]]]),
                [2, 16],
            ),
            (
                16,
                {"sequence_combiner": "mean"},
                tf.sparse.from_dense(tf.constant([[[1], [2], [3]]])),
                [1, 16],
            ),
            (
                16,
                {"sequence_combiner": "mean"},
                tf.constant([[[1], [2], [3]], [[4], [5], [6]]]),
                [2, 16],
            ),
            (12, {}, {"item_id": tf.constant([[1]])}, {"item_id": [1, 12]}),
        ],
    )
    def test_layer(self, dim, kwargs, inputs, expected_output_shape):
        column_schema = self.sample_column_schema
        layer = mm.EmbeddingTable(dim, column_schema, **kwargs)

        output = layer(inputs)
        assert_output_shape(output, expected_output_shape)

        if "sequence_combiner" in kwargs:
            assert isinstance(output, tf.Tensor)
        elif isinstance(inputs, dict):
            assert isinstance(inputs[column_schema.name], type(output[column_schema.name]))
        else:
            assert type(inputs) is type(output)

        layer_config = layer.get_config()
        copied_layer = mm.EmbeddingTable.from_config(layer_config)
        assert copied_layer.dim == layer.dim
        assert copied_layer.input_dim == layer.input_dim

        output = copied_layer(inputs)
        assert_output_shape(output, expected_output_shape)

    def test_layer_simple(self):
        col_schema = self.sample_column_schema
        dim = np.random.randint(1, high=32)
        testing_utils.layer_test(
            mm.EmbeddingTable,
            args=[dim, col_schema],
            input_data=tf.constant([[1], [2], [3]], dtype=tf.int32),
            expected_output_shape=tf.TensorShape([None, dim]),
            expected_output_dtype=tf.float32,
            supports_masking=True,
        )

    @pytest.mark.parametrize(
        ["input_shape", "expected_output_shape", "kwargs"],
        [
            (tf.TensorShape([1, 1]), tf.TensorShape([1, 10]), {}),
            (tf.TensorShape([1, 3]), tf.TensorShape([1, 3, 10]), {}),
            (tf.TensorShape([2, None]), tf.TensorShape([2, None, 10]), {}),
            (tf.TensorShape([2, None]), tf.TensorShape([2, 10]), {"sequence_combiner": "mean"}),
            ({"item_id": tf.TensorShape([1, 1])}, {"item_id": tf.TensorShape([1, 10])}, {}),
        ],
    )
    def test_compute_output_shape(self, input_shape, expected_output_shape, kwargs):
        column_schema = self.sample_column_schema
        layer = mm.EmbeddingTable(10, column_schema, **kwargs)
        output_shape = layer.compute_output_shape(input_shape)
        assert_output_shape(output_shape, expected_output_shape)

    def test_dense_with_combiner(self):
        dim = 16
        column_schema = self.sample_column_schema
        layer = mm.EmbeddingTable(dim, column_schema, sequence_combiner="mean")

        inputs = tf.constant([1])
        outputs = layer(inputs)

        assert outputs.shape == tf.TensorShape([dim])

    def test_sparse_without_combiner(self):
        dim = 16
        column_schema = self.sample_column_schema
        layer = mm.EmbeddingTable(dim, column_schema)

        inputs = tf.sparse.from_dense(tf.constant([[1, 2, 3]]))
        with pytest.raises(ValueError) as exc_info:
            layer(inputs)

        assert "Sparse tensors are not supported without sequence_combiner" in str(exc_info.value)

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

    def test_multiple_features(self):
        dim = 4
        col_schema_a = ColumnSchema(
            "a", dtype=np.int32, properties={"domain": {"min": 0, "max": 10}}
        )
        col_schema_b = ColumnSchema(
            "b", dtype=np.int32, properties={"domain": {"min": 0, "max": 10}}
        )
        col_schema_c = ColumnSchema(
            "c", dtype=np.int32, properties={"domain": {"min": 0, "max": 10}}
        )

        embedding_table = mm.EmbeddingTable(dim, col_schema_a, col_schema_b)
        embedding_table.add_feature(col_schema_c)

        assert embedding_table.schema == Schema([col_schema_a, col_schema_b, col_schema_c])

        result = embedding_table(
            {"a": tf.constant([[[1]]]), "b": tf.ragged.constant([[[1]]]), "c": tf.constant([1])}
        )

        assert set(result.keys()) == {"a", "b", "c"}
        np.testing.assert_array_equal(result["a"].numpy(), result["b"].numpy())

    def test_incompatible_features(self):
        dim = 4
        col_schema_a = ColumnSchema(
            "a", dtype=np.int32, properties={"domain": {"min": 0, "max": 10}}
        )
        col_schema_b = ColumnSchema(
            "b", dtype=np.int32, properties={"domain": {"min": 0, "max": 20}}
        )

        with pytest.raises(ValueError) as exc_info:
            mm.EmbeddingTable(dim, col_schema_a, col_schema_b)
        assert "does not match existing input dim" in str(exc_info.value)

    def test_select_by_tag(self):
        dim = 4

        col_schema_a = ColumnSchema(
            "a",
            dtype=np.int32,
            properties={"domain": {"min": 0, "max": 10}},
            tags=[Tags.USER, Tags.CATEGORICAL],
        )
        col_schema_b = ColumnSchema(
            "b",
            dtype=np.int32,
            properties={"domain": {"min": 0, "max": 10}},
            tags=[Tags.USER, Tags.CATEGORICAL],
        )
        col_schema_c = ColumnSchema(
            "c",
            dtype=np.int32,
            properties={"domain": {"min": 0, "max": 10}},
            tags=[Tags.ITEM, Tags.CATEGORICAL],
        )

        embedding_table = mm.EmbeddingTable(dim, col_schema_a, col_schema_b, col_schema_c)

        categorical = embedding_table.select_by_tag(Tags.CATEGORICAL)
        assert isinstance(categorical, mm.EmbeddingTable)
        assert sorted(categorical.features) == ["a", "b", "c"]

        inputs = {
            "a": tf.constant([[0], [1], [2]]),
            "b": tf.constant([[3], [4], [5]]),
            "c": tf.constant([[6], [7], [8]]),
        }

        _ = categorical(inputs)
        _ = embedding_table(inputs)

        assert np.allclose(
            categorical.table.embeddings.numpy(), embedding_table.table.embeddings.numpy()
        )
        assert categorical.table is embedding_table.table

        assert embedding_table.select_by_tag(Tags.CONTINUOUS) is None

        user = embedding_table.select_by_tag(Tags.USER)
        assert isinstance(user, mm.EmbeddingTable)
        assert sorted(user.features) == ["a", "b"]

        _ = user(inputs)
        assert np.allclose(
            categorical.table.embeddings.numpy(), embedding_table.table.embeddings.numpy()
        )
        assert user.table is embedding_table.table

        item = embedding_table.select_by_tag(Tags.ITEM)
        assert isinstance(item, mm.EmbeddingTable)
        assert sorted(item.features) == ["c"]
        assert item.table is embedding_table.table


@pytest.mark.parametrize("trainable", [True, False])
def test_pretrained_from_InputBlockV2(trainable, music_streaming_data: Dataset):
    vocab_size = music_streaming_data.schema.column_schemas["item_id"].int_domain.max + 1
    embedding_dim = 32
    weights = np.random.rand(vocab_size, embedding_dim)
    pre_trained_weights_df = pd.DataFrame(weights)

    embed_dims = {}
    embed_dims["item_id"] = pre_trained_weights_df.shape[1]
    embeddings_init = {
        "item_id": mm.TensorInitializer(weights),
    }

    embeddings_block = mm.Embeddings(
        music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL),
        embeddings_initializer=embeddings_init,
        trainable={"item_id": trainable},
        dim=embed_dims,
    )
    input_block = mm.InputBlockV2(music_streaming_data.schema, categorical=embeddings_block)

    model = mm.DCNModel(
        music_streaming_data.schema,
        depth=2,
        input_block=input_block,
        deep_block=mm.MLPBlock([64, 32]),
        prediction_tasks=mm.BinaryClassificationTask("click"),
    )
    model_test(model, music_streaming_data)

    if trainable:
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            weights,
            embeddings_block["item_id"].table.embeddings,
        )
    else:
        np.testing.assert_array_almost_equal(
            weights,
            embeddings_block["item_id"].table.embeddings,
        )


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


def test_embeddings_with_regularization(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.ITEM_ID)
    dim = 16
    embeddings_wo_reg = mm.Embeddings(schema, dim=dim)
    embeddings_batch_reg = mm.Embeddings(schema, dim=dim, l2_batch_regularization_factor=0.2)
    embeddings_table_reg = mm.Embeddings(
        schema, dim=dim, embeddings_regularizer=tf.keras.regularizers.L2(0.2)
    )

    inputs = mm.sample_batch(testing_data, batch_size=100, include_targets=False)
    _ = embeddings_wo_reg(inputs)
    _ = embeddings_batch_reg(inputs)
    _ = embeddings_table_reg(inputs)

    assert not embeddings_wo_reg.losses
    assert embeddings_batch_reg.losses[0] > 0
    tf.debugging.assert_greater(embeddings_table_reg.losses[0], embeddings_batch_reg.losses[0])


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


def test_embedding_features_from_config():
    schema = Schema(
        [
            ColumnSchema(
                "name",
                tags=[Tags.USER, Tags.CATEGORICAL],
                is_list=False,
                is_ragged=False,
                dtype=np.int32,
                properties={
                    "num_buckets": None,
                    "freq_threshold": 0,
                    "max_size": 0,
                    "start_index": 0,
                    "cat_path": ".//categories/unique.name.parquet",
                    "domain": {"min": 0, "max": 5936, "name": "name"},
                    "embedding_sizes": {"cardinality": 5937, "dimension": 208},
                },
            ),
            ColumnSchema(
                "feature",
                tags=[Tags.USER, Tags.CATEGORICAL],
                is_list=False,
                is_ragged=False,
                dtype=np.int32,
                properties={
                    "num_buckets": None,
                    "freq_threshold": 0,
                    "max_size": 0,
                    "start_index": 0,
                    "cat_path": ".//categories/unique.feature.parquet",
                    "domain": {"min": 0, "max": 2, "name": "feature"},
                    "embedding_sizes": {"cardinality": 3, "dimension": 16},
                },
            ),
        ]
    )

    embedding_features = mm.EmbeddingFeatures.from_schema(
        schema,
        tags=(Tags.CATEGORICAL,),
        embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
    )
    config = embedding_features.get_config()
    reloaded_embedding_features = mm.EmbeddingFeatures.from_config(config)

    assert set(embedding_features.embedding_tables.keys()) == set(
        reloaded_embedding_features.embedding_tables.keys()
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
    if cudf:
        items_embeddings_dataset = emb_module.embedding_table_dataset(Tags.ITEM_ID, gpu=True)
        assert np.allclose(
            item_id_embeddings.numpy(),
            items_embeddings_dataset.to_ddf().compute().to_pandas().values,
        )

        emb_init = mm.TensorInitializer.from_dataset(items_embeddings_dataset)
        assert np.allclose(item_id_embeddings.numpy(), emb_init(item_id_embeddings.shape).numpy())


def test_shared_embeddings(music_streaming_data: Dataset):
    inputs = mm.InputBlock(music_streaming_data.schema)

    embeddings = inputs.select_by_name(Tags.CATEGORICAL.value)

    assert embeddings.table_config("item_genres") == embeddings.table_config("user_genres")


class TestEmbeddings:
    def test_shared_domain(self):
        schema = Schema(
            [
                ColumnSchema(
                    "item_id",
                    dtype=np.int32,
                    properties={"domain": {"min": 0, "max": 10, "name": "item_id_embedding"}},
                ),
                ColumnSchema(
                    "user_item_history",
                    dtype=np.int32,
                    properties={"domain": {"min": 0, "max": 10, "name": "item_id_embedding"}},
                ),
                ColumnSchema(
                    "item_feature_a", dtype=np.int32, properties={"domain": {"min": 0, "max": 20}}
                ),
            ]
        )
        embeddings = mm.Embeddings(schema)
        assert {layer.name for layer in embeddings.layers} == {
            "item_id_embedding",
            "item_feature_a",
        }
        assert set(embeddings.parallel_layers.keys()) == {"item_id_embedding", "item_feature_a"}

        outputs = embeddings(
            {
                "item_id": tf.constant([[1]]),
                "user_item_history": tf.ragged.constant([[[1]], [[2], [3]]]),
                "item_feature_a": tf.constant([[2]]),
            }
        )

        assert set(outputs.keys()) == {"item_id", "user_item_history", "item_feature_a"}
        np.testing.assert_array_equal(outputs["item_id"], outputs["user_item_history"][:1])
