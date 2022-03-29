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
from merlin.models.data.synthetic import SyntheticData
from merlin.models.tf.inputs.embedding import EmbeddingTable
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_embedding_features(tf_cat_features):
    options = ml.EmbeddingTableOptions(dim=64)
    embedding_tables = {f: EmbeddingTable(f, 100, options) for f in tf_cat_features.keys()}
    embeddings = ml.EmbeddingFeatures(embedding_tables)(tf_cat_features)

    assert list(embeddings.keys()) == list(embedding_tables.keys())
    assert all([emb.shape[-1] == options.dim for emb in embeddings.values()])


def test_embedding_features_yoochoose(testing_data: SyntheticData):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = ml.EmbeddingFeatures.from_schema(schema)
    embeddings = emb_module(testing_data.tf_tensor_dict)

    assert sorted(list(embeddings.keys())) == sorted(schema.column_names)
    assert all(emb.shape[-1] == 64 for emb in embeddings.values())
    max_value = list(schema.select_by_name("item_id"))[0].int_domain.max
    assert emb_module["item_id"].table.shape[0] == max_value + 1


def test_serialization_embedding_features(testing_data: SyntheticData):
    inputs = ml.EmbeddingFeatures.from_schema(testing_data.schema)

    copy_layer = testing_utils.assert_serialization(inputs)

    assert list(inputs.embeddings.keys()) == list(copy_layer.embeddings.keys())


@testing_utils.mark_run_eagerly_modes
def test_embedding_features_yoochoose_model(testing_data: SyntheticData, run_eagerly):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    inputs = ml.EmbeddingFeatures(ml.EmbeddingOptions(schema), aggregation="concat")
    body = ml.SequentialBlock([inputs, ml.MLPBlock([64])])

    testing_utils.assert_body_works_in_model(testing_data.tf_tensor_dict, inputs, body, run_eagerly)


def test_embedding_features_yoochoose_custom_dims(testing_data: SyntheticData):
    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    emb_module = ml.EmbeddingFeatures(
        ml.EmbeddingOptions(
            schema,
            custom_tables={"item_id": ml.EmbeddingTableOptions(dim=100)},
            default_embedding_dim=64,
        ),
    )

    embeddings = emb_module(testing_data.tf_tensor_dict)

    assert emb_module["item_id"].table.shape[1] == 100
    assert emb_module["categories"].table.shape[1] == 64

    assert embeddings["item_id"].shape[1] == 100
    assert embeddings["categories"].shape[1] == 64


def test_embedding_features_yoochoose_infer_embedding_sizes(testing_data: SyntheticData):
    emb_module = ml.EmbeddingFeatures(
        ml.EmbeddingOptions(
            testing_data.schema.select_by_tag(Tags.CATEGORICAL),
            infer_embedding_sizes=True,
            infer_embedding_sizes_multiplier=3.0,
        ),
    )

    embeddings = emb_module(testing_data.tf_tensor_dict)

    assert emb_module["item_id"].table.shape[1] == 46
    assert emb_module["categories"].table.shape[1] == 13

    assert embeddings["item_id"].shape[1] == 46
    assert embeddings["categories"].shape[1] == 13


def test_embedding_features_yoochoose_custom_initializers(testing_data: SyntheticData):
    item_initializer = init_ops_v2.TruncatedNormal(mean=1.0, stddev=0.05)
    category_initializer = init_ops_v2.TruncatedNormal(mean=2.0, stddev=0.1)

    schema = testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    emb_module = ml.EmbeddingFeatures(
        ml.EmbeddingOptions(
            schema,
            custom_tables={
                "item_id": ml.EmbeddingTableOptions(64, initializer=item_initializer),
                "categories": ml.EmbeddingTableOptions(64, initializer=category_initializer),
            },
        ),
    )

    embeddings = emb_module(testing_data.tf_tensor_dict)

    assert embeddings["item_id"].numpy().mean() == pytest.approx(item_initializer.mean, abs=0.1)
    assert embeddings["item_id"].numpy().std() == pytest.approx(item_initializer.stddev, abs=0.1)

    assert embeddings["categories"].numpy().mean() == pytest.approx(
        category_initializer.mean, abs=0.1
    )
    assert embeddings["categories"].numpy().std() == pytest.approx(
        category_initializer.stddev, abs=0.1
    )


def test_shared_embeddings(music_streaming_data: SyntheticData):
    embeddings = ml.EmbeddingFeatures(ml.EmbeddingOptions(music_streaming_data.schema))

    assert embeddings["item_genres"] == embeddings["user_genres"]
