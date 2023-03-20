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
import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import ColumnSchema, Schema, Tags


def test_tabular_features(testing_data: Dataset):
    tab_module = mm.InputBlock(testing_data.schema)

    outputs = tab_module(mm.sample_batch(testing_data, batch_size=100, include_targets=False))

    con = testing_data.schema.select_by_tag(Tags.CONTINUOUS).column_names
    cat = testing_data.schema.select_by_tag(Tags.CATEGORICAL).column_names

    assert set(outputs.keys()) == set(con + cat)


def test_serialization_tabular_features(testing_data: Dataset):
    inputs = mm.InputBlock(testing_data.schema)

    copy_layer = testing_utils.assert_serialization(inputs)

    assert list(inputs.parallel_layers.keys()) == list(copy_layer.parallel_layers.keys())


def test_tabular_features_with_projection(testing_data: Dataset):
    tab_module = mm.InputBlock(testing_data.schema, continuous_projection=mm.MLPBlock([64]))

    outputs = tab_module(mm.sample_batch(testing_data, batch_size=100, include_targets=False))
    continuous_feature_names = testing_data.schema.select_by_tag(Tags.CONTINUOUS).column_names

    assert len(set(continuous_feature_names).intersection(set(outputs.keys()))) == 0
    assert "continuous_projection" in outputs
    assert list(outputs["continuous_projection"].shape)[1] == 64


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize("continuous_projection", [None, 128])
def test_tabular_features_yoochoose_model(
    music_streaming_data: Dataset, run_eagerly, continuous_projection
):
    if continuous_projection:
        continuous_projection = mm.MLPBlock([continuous_projection])
    inputs = mm.InputBlock(
        music_streaming_data.schema,
        continuous_projection=continuous_projection,
        aggregation="concat",
    )

    body = mm.SequentialBlock([inputs, mm.MLPBlock([64])])
    model = mm.Model(body, mm.BinaryClassificationTask("click"))

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


@testing_utils.mark_run_eagerly_modes
@pytest.mark.parametrize("continuous_projection", [None, 128])
def test_tabular_features_yoochoose_model_inputblockv2(
    music_streaming_data: Dataset, run_eagerly, continuous_projection
):
    kwargs = {}
    if continuous_projection:
        kwargs["continuous"] = mm.ContinuousProjection(
            music_streaming_data.schema.select_by_tag(Tags.CONTINUOUS),
            mm.MLPBlock([continuous_projection]),
        )

    inputs = mm.InputBlockV2(music_streaming_data.schema, aggregation="concat", **kwargs)

    body = mm.SequentialBlock([inputs, mm.MLPBlock([64])])
    model = mm.Model(body, mm.BinaryClassificationTask("click"))

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


def test_tabular_seq_features_ragged_embeddings(sequence_testing_data: Dataset):
    tab_module = mm.InputBlockV2(
        sequence_testing_data.schema,
        categorical=mm.Embeddings(
            sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL), sequence_combiner=None
        ),
        aggregation=None,
    )

    loader = mm.Loader(sequence_testing_data, batch_size=100)
    batch = mm.sample_batch(loader, include_targets=False, prepare_features=True)

    outputs = tab_module(batch)

    con = sequence_testing_data.schema.select_by_tag(Tags.CONTINUOUS).column_names
    cat = sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL).column_names
    seq = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE).column_names

    assert set(outputs.keys()) == set(con + cat)
    assert all(isinstance(val, tf.RaggedTensor) for name, val in outputs.items() if name in seq)


@pytest.mark.parametrize(
    "seq_combiner",
    [tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)), "mean"],
)
def test_tabular_seq_features_ragged_emb_combiner(sequence_testing_data: Dataset, seq_combiner):
    con2d = sequence_testing_data.schema.select_by_tag(Tags.CONTINUOUS).remove_by_tag(Tags.SEQUENCE)
    input_block = mm.InputBlockV2(
        sequence_testing_data.schema,
        categorical=mm.Embeddings(
            sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL),
            sequence_combiner=seq_combiner,
        ),
        continuous=mm.Continuous(con2d),
        aggregation=None,
    )

    loader = mm.Loader(sequence_testing_data, batch_size=100)
    batch = mm.sample_batch(loader, include_targets=False, prepare_features=True)

    outputs = input_block(batch)

    cat = sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL).column_names

    assert all(isinstance(val, tf.Tensor) for name, val in outputs.items())
    assert all(tf.rank(val) == 2 for name, val in outputs.items() if name in cat)
    assert set(cat + con2d.column_names) == set(outputs.keys())


def test_tabular_seq_features_ragged_custom_emb_combiner(sequence_testing_data: Dataset):
    schema = sequence_testing_data.schema
    schema = schema + Schema([ColumnSchema("item_id_seq_weights")])
    assert "item_id_seq_weights" in schema.column_names

    loader = mm.Loader(sequence_testing_data, batch_size=100)
    batch = mm.sample_batch(loader, include_targets=False, prepare_features=True)
    batch["item_id_seq_weights"] = tf.ragged.constant(
        [[1.0, 2.0, 3.0, 4.0] for _ in range(batch["item_id_seq"].shape[0])],
        row_splits_dtype=batch["item_id_seq"].row_splits.dtype,
    )

    input_block_weighed_avg = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            schema.select_by_tag(Tags.CATEGORICAL),
            sequence_combiner=mm.AverageEmbeddingsByWeightFeature.from_schema_convention(
                schema, "_weights"
            ),
        ),
        aggregation=None,
    )

    outputs_weighted_avg = input_block_weighed_avg(batch, features=batch)

    input_block_simple_avg = mm.InputBlockV2(
        schema,
        categorical=mm.Embeddings(
            schema.select_by_tag(Tags.CATEGORICAL),
            sequence_combiner=tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),
        ),
        aggregation=None,
    )

    outputs_simple_avg = input_block_simple_avg(batch, features=batch)

    assert not tf.reduce_all(
        outputs_weighted_avg["item_id_seq"] == outputs_simple_avg["item_id_seq"]
    )

    cat = schema.select_by_tag(Tags.CATEGORICAL).column_names

    assert all(
        isinstance(val, tf.Tensor) for name, val in outputs_weighted_avg.items() if name in cat
    )
    assert all(tf.rank(val) == 2 for name, val in outputs_weighted_avg.items() if name in cat)


def test_tabular_seq_features_avg_embeddings_with_mapvalues(sequence_testing_data: Dataset):
    cat_schema = sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL)

    batch = mm.sample_batch(
        sequence_testing_data, batch_size=100, include_targets=False, prepare_features=True
    )

    input_block = mm.InputBlockV2(
        cat_schema,
        categorical=mm.Embeddings(
            cat_schema,
        ),
        post=mm.MapValues(
            tf.keras.layers.Lambda(
                lambda x: tf.math.reduce_mean(x, axis=1) if isinstance(x, tf.RaggedTensor) else x
            )
        ),
        aggregation=None,
    )

    output = input_block(batch)
    assert all(
        isinstance(val, tf.Tensor)
        for name, val in output.items()
        if name in cat_schema.column_names
    )
    assert all(tf.rank(val) == 2 for name, val in output.items() if name in cat_schema.column_names)


@pytest.mark.parametrize("aggregation", [None, "concat"])
def test_embedding_tables_from_schema_infer_dims(sequence_testing_data: Dataset, aggregation: str):
    cat_schema = sequence_testing_data.schema.select_by_tag(Tags.CATEGORICAL)
    embeddings_block = mm.Embeddings(
        cat_schema.select_by_tag(Tags.CATEGORICAL),
        dim={"item_id_seq": 15, "test_user_id": 21},
        embeddings_initializer="truncated_normal",
    )
    input_block = mm.InputBlockV2(cat_schema, categorical=embeddings_block, aggregation=aggregation)

    loader = mm.Loader(sequence_testing_data, batch_size=100)
    batch = mm.sample_batch(loader, include_targets=False, prepare_features=True)

    outputs = input_block(batch)

    if aggregation == "concat":
        assert outputs.shape[-1] == 60
    elif aggregation is None:
        assert {k: v.shape[-1] for k, v in outputs.items()} == {
            "test_user_id": 21,
            "item_id_seq": 15,
            # Inferred dims from cardinality
            "categories": 16,
            "user_country": 8,
        }
