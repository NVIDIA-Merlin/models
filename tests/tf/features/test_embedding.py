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

import pytest
from tensorflow.python.ops import init_ops_v2

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_embedding_features(tf_cat_features):
    dim = 15
    feature_config = {
        f: ml.FeatureConfig(ml.TableConfig(100, dim, name=f, initializer=None))
        for f in tf_cat_features.keys()
    }
    embeddings = ml.EmbeddingFeatures(feature_config)(tf_cat_features)

    assert list(embeddings.keys()) == list(feature_config.keys())
    assert all([emb.shape[-1] == dim for emb in embeddings.values()])


def test_embedding_features_yoochoose(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = ml.EmbeddingFeatures.from_schema(schema)
    embeddings = emb_module(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert sorted(list(embeddings.keys())) == sorted(schema.column_names)
    assert all(emb.shape[-1] == 64 for emb in embeddings.values())
    max_value = list(schema.select_by_name("item_id"))[0].int_domain.max
    assert emb_module.embedding_tables["item_id"].shape[0] == max_value + 1


def test_serialization_embedding_features(testing_data: Dataset):
    inputs = ml.EmbeddingFeatures.from_schema(testing_data.schema)

    copy_layer = testing_utils.assert_serialization(inputs)

    assert list(inputs.feature_config.keys()) == list(copy_layer.feature_config.keys())

    from merlin.models.tf.features.embedding import serialize_table_config as serialize

    assert all(
        serialize(inputs.feature_config[key].table)
        == serialize(copy_layer.feature_config[key].table)
        for key in copy_layer.feature_config
    )


@testing_utils.mark_run_eagerly_modes
def test_embedding_features_yoochoose_model(music_streaming_data: Dataset, run_eagerly):
    schema = music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL)

    inputs = ml.EmbeddingFeatures.from_schema(schema, aggregation="concat")
    body = ml.SequentialBlock([inputs, ml.MLPBlock([64])])

    testing_utils.assert_body_works_in_model(music_streaming_data, body, run_eagerly)


def test_embedding_features_yoochoose_custom_dims(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = ml.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=ml.EmbeddingOptions(
            embedding_dims={"item_id": 100}, embedding_dim_default=64
        ),
    )

    embeddings = emb_module(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert emb_module.embedding_tables["item_id"].shape[1] == 100
    assert emb_module.embedding_tables["categories"].shape[1] == 64

    assert embeddings["item_id"].shape[1] == 100
    assert embeddings["categories"].shape[1] == 64


def test_embedding_features_yoochoose_infer_embedding_sizes(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = ml.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=ml.EmbeddingOptions(
            infer_embedding_sizes=True, infer_embedding_sizes_multiplier=3.0
        ),
    )

    embeddings = emb_module(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert emb_module.embedding_tables["item_id"].shape[1] == 46
    assert emb_module.embedding_tables["categories"].shape[1] == 13

    assert embeddings["item_id"].shape[1] == 46
    assert embeddings["categories"].shape[1] == 13


def test_embedding_features_yoochoose_custom_initializers(testing_data: Dataset):
    ITEM_MEAN = 1.0
    ITEM_STD = 0.05

    CATEGORY_MEAN = 2.0
    CATEGORY_STD = 0.1

    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    emb_module = ml.EmbeddingFeatures.from_schema(
        schema,
        embedding_options=ml.EmbeddingOptions(
            embeddings_initializers={
                "item_id": init_ops_v2.TruncatedNormal(mean=ITEM_MEAN, stddev=ITEM_STD),
                "categories": init_ops_v2.TruncatedNormal(mean=CATEGORY_MEAN, stddev=CATEGORY_STD),
            },
        ),
    )

    embeddings = emb_module(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert embeddings["item_id"].numpy().mean() == pytest.approx(ITEM_MEAN, abs=0.1)
    assert embeddings["item_id"].numpy().std() == pytest.approx(ITEM_STD, abs=0.1)

    assert embeddings["categories"].numpy().mean() == pytest.approx(CATEGORY_MEAN, abs=0.1)
    assert embeddings["categories"].numpy().std() == pytest.approx(CATEGORY_STD, abs=0.1)


def test_shared_embeddings(music_streaming_data: Dataset):
    inputs = ml.InputBlock(music_streaming_data.schema)

    embeddings = inputs.select_by_name(Tags.CATEGORICAL.value)

    assert embeddings.table_config("item_genres") == embeddings.table_config("user_genres")
