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
import pytest
from tensorflow.keras.initializers import RandomUniform

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_embedding_features(tf_cat_features):
    dim = 15
    feature_config = {
        f: mm.FeatureConfig(mm.TableConfig(100, dim, name=f, initializer=None))
        for f in tf_cat_features.keys()
    }
    embeddings = mm.EmbeddingFeatures(feature_config)(tf_cat_features)

    assert list(embeddings.keys()) == list(feature_config.keys())
    assert all([emb.shape[-1] == dim for emb in embeddings.values()])


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
    assert emb_module.embedding_tables["item_id"].shape[0] == max_value + 1

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

    assert emb_module.embedding_tables["item_id"].shape[1] == 100
    assert emb_module.embedding_tables["categories"].shape[1] == 64

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

    assert emb_module.embedding_tables["user_id"].shape[1] == embeddings["user_id"].shape[1] == 20
    assert (
        emb_module.embedding_tables["user_country"].shape[1]
        == embeddings["user_country"].shape[1]
        == 9
    )
    assert emb_module.embedding_tables["item_id"].shape[1] == embeddings["item_id"].shape[1] == 46
    assert (
        emb_module.embedding_tables["categories"].shape[1]
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

    assert emb_module.embedding_tables["user_id"].shape[1] == embeddings["user_id"].shape[1] == 24
    assert (
        emb_module.embedding_tables["user_country"].shape[1]
        == embeddings["user_country"].shape[1]
        == 16
    )
    assert emb_module.embedding_tables["item_id"].shape[1] == embeddings["item_id"].shape[1] == 48
    assert (
        emb_module.embedding_tables["categories"].shape[1]
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

    assert emb_module.embedding_tables["user_id"].shape[1] == embeddings["user_id"].shape[1] == 50
    assert (
        emb_module.embedding_tables["user_country"].shape[1]
        == embeddings["user_country"].shape[1]
        == 100
    )
    assert emb_module.embedding_tables["item_id"].shape[1] == embeddings["item_id"].shape[1] == 46
    assert (
        emb_module.embedding_tables["categories"].shape[1]
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

    assert np.allclose(emb_module.embedding_tables["item_id"].numpy(), pretrained_emb_item_ids)
    assert np.allclose(emb_module.embedding_tables["categories"].numpy(), pretrained_emb_categories)


def test_embedding_features_exporting_and_loading_pretrained_initializer(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    emb_module = mm.EmbeddingFeatures.from_schema(schema)

    # Calling the first batch, so that embedding tables are build
    _ = emb_module(mm.sample_batch(testing_data, batch_size=10, include_targets=False))
    item_id_embeddings = emb_module.embedding_tables["item_id"]

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
