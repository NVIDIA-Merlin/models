import numpy as np
import pytest
import tensorflow as tf
from tensorflow.test import TestCase

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import ColumnSchema, Schema, Tags


@pytest.mark.parametrize("name_branches", [True, False])
def test_parallel_block_pruning(music_streaming_data: Dataset, name_branches: bool):
    music_streaming_data.schema = music_streaming_data.schema.remove_by_tag(Tags.CONTINUOUS)

    continuous_block = mm.Filter(music_streaming_data.schema.select_by_tag(Tags.CONTINUOUS))
    embedding_block = mm.Embeddings(music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL))

    if name_branches:
        branches = {"continuous": continuous_block, "embeddings": embedding_block}
    else:
        branches = [continuous_block, embedding_block]

    input_block = mm.ParallelBlock(branches, schema=music_streaming_data.schema)

    features = mm.sample_batch(music_streaming_data, batch_size=10, include_targets=False)

    outputs = input_block(features)

    assert len(outputs) == 7  # There are 7 categorical features
    assert continuous_block not in input_block.parallel_values


def test_parallel_block_serialization(music_streaming_data: Dataset):
    unknown_filter = mm.Filter(["none"])
    block = mm.ParallelBlock(mm.Filter(["position"]), unknown_filter, automatic_pruning=False)
    block_copy = block.from_config(block.get_config())

    assert not block_copy.automatic_pruning
    assert unknown_filter not in block_copy.parallel_values

    features = mm.sample_batch(music_streaming_data, batch_size=10, include_targets=False)

    outputs_1 = block(features)
    outputs_2 = block_copy(features)

    for key in outputs_1:
        np.testing.assert_array_equal(outputs_2[key].numpy(), outputs_1[key].numpy())
    assert len(outputs_1) == len(outputs_2) == 1


@pytest.mark.parametrize("name_branches", [True, False])
def test_parallel_block_schema_propagation(music_streaming_data, name_branches: bool):
    continuous_block = mm.Filter(Tags.CONTINUOUS)
    embedding_block = mm.Embeddings(music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL))

    if name_branches:
        branches = {"continuous": continuous_block, "embeddings": embedding_block}
    else:
        branches = [continuous_block, embedding_block]

    input_block = mm.ParallelBlock(branches, schema=music_streaming_data.schema)
    features = mm.sample_batch(music_streaming_data, batch_size=10, include_targets=False)
    outputs = input_block(features)

    assert len(outputs) == 10  # There are 7 categorical + 3 continuous features


@pytest.mark.parametrize("name_branches", [True, False])
def test_parallel_block_with_layers(music_streaming_data, name_branches: bool):
    d, d_1 = tf.keras.layers.Dense(32), tf.keras.layers.Dense(32)
    if name_branches:
        branches = {"d": d, "d_1": d_1}
    else:
        branches = [d, d_1]

    block = mm.ParallelBlock(branches, aggregation="concat")
    model = mm.Model.from_block(block, music_streaming_data.schema)

    outputs = block(tf.constant([[2.0]]))
    assert outputs.shape == tf.TensorShape([1, 64])

    features = mm.sample_batch(music_streaming_data, batch_size=10, include_targets=False)
    outputs = model(features)

    assert len(outputs) == 3  # number of prediction tasks in this schema


def test_parallel_block_select_from_names():
    test_case = TestCase()
    branches = {
        "1": tf.keras.layers.Dense(32),
        "2": tf.keras.layers.Dense(32),
        "3": tf.keras.layers.Dense(32),
    }
    blocks = mm.ParallelBlock(branches, aggregation="concat")
    assert 2 == len(blocks.select_by_names(["1", "2"]))
    with test_case.assertRaisesRegex(ValueError, "is not in ParallelBlock"):
        blocks.select_by_names(["0", "2"])


def test_parallel_block_select_by_tags(music_streaming_data):
    continuous_block = mm.Filter(Tags.CONTINUOUS)
    embedding_block = mm.Embeddings(
        schema=music_streaming_data.schema.select_by_tag(Tags.CATEGORICAL)
    )
    branches = {"continuous": continuous_block, "embeddings": embedding_block}
    parallel_block = mm.ParallelBlock(branches, schema=music_streaming_data.schema)

    continuous_inputs = parallel_block.select_by_tag(Tags.CONTINUOUS)
    assert isinstance(continuous_inputs, mm.ParallelBlock)
    assert sorted(continuous_inputs.schema.column_names) == [
        "item_recency",
        "position",
        "user_age",
    ]

    categorical_inputs = parallel_block.select_by_tag(Tags.CATEGORICAL)
    assert sorted(categorical_inputs.schema.column_names) == [
        "country",
        "item_category",
        "item_genres",
        "item_id",
        "session_id",
        "user_genres",
        "user_id",
    ]

    # Forward pass a batch and compare the input features to output features
    batch = mm.sample_batch(music_streaming_data, batch_size=10, include_targets=False)
    continuous_outputs = continuous_inputs(batch)
    assert sorted(continuous_outputs.keys()) == sorted(continuous_inputs.schema.column_names)
    categorical_outputs = categorical_inputs(batch)
    assert sorted(categorical_outputs.keys()) == sorted(categorical_inputs.schema.column_names)

    # There are no sequence or time features in the dataset.
    assert parallel_block.select_by_tag([Tags.SEQUENCE, Tags.TIME]) is None

    # InputBlock is also a ParallelBlock.
    input_block = mm.InputBlockV2(music_streaming_data.schema, aggregation=None)
    item_inputs = input_block.select_by_tag(Tags.ITEM)
    assert sorted(item_inputs.schema.column_names) == [
        "item_category",
        "item_genres",
        "item_id",
        "item_recency",
    ]
    outputs = item_inputs(batch)
    assert sorted(outputs.keys()) == sorted(item_inputs.schema.column_names)


def test_parallel_block_select_by_tag_args():
    is_input = True
    pre = tf.keras.layers.Dense(5, name="pre")
    post = tf.keras.layers.Dense(5, name="post")
    aggregation = mm.ConcatFeatures(name="agg")

    user_schema = Schema([ColumnSchema("user_feature", tags=[Tags.USER])])
    item_schema = Schema([ColumnSchema("item_feature", tags=[Tags.ITEM])])
    schema = user_schema + item_schema

    branches = {
        "user": mm.ParallelBlock([tf.keras.layers.Dense(10)], schema=user_schema),
        "item": mm.ParallelBlock([tf.keras.layers.Dense(10)], schema=item_schema),
    }

    parallel_block = mm.ParallelBlock(
        branches, schema=schema, is_input=is_input, pre=pre, post=post, aggregation=aggregation
    )

    user_block = parallel_block.select_by_tag(Tags.USER)
    item_block = parallel_block.select_by_tag(Tags.ITEM)

    assert user_block.is_input == is_input == item_block.is_input
    assert user_block.pre == pre == item_block.pre
    assert user_block.post == post == item_block.post
    assert user_block.aggregation == aggregation == item_block.aggregation


class TestCond:
    def test_true(self):
        condition = tf.keras.layers.Lambda(lambda _: True)
        true = tf.keras.layers.Lambda(lambda _: tf.constant([4.0]))
        false = tf.keras.layers.Lambda(lambda _: tf.constant([2.0]))
        output_data = testing_utils.layer_test(
            mm.Cond, kwargs=dict(condition=condition, true=true, false=false), input_shape=(1,)
        )
        np.testing.assert_array_equal(output_data, np.array([4.0]))

    def test_false(self):
        condition = tf.keras.layers.Lambda(lambda _: False)
        true = tf.keras.layers.Lambda(lambda _: tf.constant([4.0]))
        false = tf.keras.layers.Lambda(lambda _: tf.constant([2.0]))
        output_data = testing_utils.layer_test(
            mm.Cond, kwargs=dict(condition=condition, true=true, false=false), input_shape=(1,)
        )
        np.testing.assert_array_equal(output_data, np.array([2.0]))

    def test_divide(self):
        condition = tf.keras.layers.Lambda(lambda _: True)
        true = tf.keras.layers.Lambda(lambda x: x / 2)
        false = tf.keras.layers.Lambda(lambda x: x / 5)
        output_data = testing_utils.layer_test(
            mm.Cond,
            kwargs=dict(condition=condition, true=true, false=false),
            input_data=tf.convert_to_tensor([np.arange(5).astype(np.float32)]),
        )
        np.testing.assert_array_equal(output_data, np.array([[0.0, 0.5, 1.0, 1.5, 2.0]]))

    def test_default_false(self):
        condition = tf.keras.layers.Lambda(lambda _: False)
        true = tf.keras.layers.Lambda(lambda _: tf.constant([4.0]))
        output_data = testing_utils.layer_test(
            mm.Cond, kwargs=dict(condition=condition, true=true), input_data=tf.constant([3.0])
        )
        np.testing.assert_array_equal(output_data, np.array([3.0]))

    def test_different_output_shapes(self):
        condition = tf.keras.layers.Lambda(lambda _: True)
        true = tf.keras.layers.Dense(1)
        false = tf.keras.layers.Dense(2)
        with pytest.raises(ValueError) as exc_info:
            testing_utils.layer_test(
                mm.Cond,
                kwargs=dict(condition=condition, true=true, false=false),
                input_data=tf.constant([[1.0]]),
            )
        assert "true and false branches must return the same output shape" in str(exc_info.value)

    def test_with_blocks(self):
        condition = tf.keras.layers.Lambda(lambda _: True)
        true = mm.MLPBlock([10])
        false = mm.SequentialBlock([mm.MLPBlock([100]), mm.MLPBlock([10])])
        output_data = testing_utils.layer_test(
            mm.Cond,
            kwargs=dict(condition=condition, true=true, false=false),
            input_data=tf.constant([[3.0]]),
        )
        assert output_data.shape == (1, 10)

    @pytest.mark.parametrize("run_eagerly", [True, False])
    def test_with_model(self, run_eagerly, music_streaming_data):
        condition = tf.keras.layers.Lambda(lambda _: tf.random.uniform((1,)) < 0.5)

        true = tf.keras.layers.Lambda(lambda x: x * 2)

        layer = mm.Cond(condition, true)

        model = mm.Model(
            tf.keras.layers.Lambda(lambda x: x["item_recency"]),
            layer,
            tf.keras.layers.Dense(1),
            mm.BinaryClassificationTask("click"),
            schema=music_streaming_data.schema.select_by_name("item_recency"),
        )

        testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)


class TestMapValues:
    def test_map_values_tensor(self):
        map_layer = tf.keras.layers.Lambda(lambda _: tf.constant([4.0]))
        output_data = testing_utils.layer_test(
            mm.MapValues, kwargs=dict(layer=map_layer), input_shape=(1,)
        )
        np.testing.assert_array_equal(output_data, np.array([4.0]))

    def test_map_values_dict(self):
        map_layer = mm.MapValues(tf.keras.layers.Lambda(lambda _: tf.constant([4.0])))

        out = map_layer({"a": tf.constant([1.0]), "b": tf.constant([2.0])})
        np.testing.assert_array_equal(out, {k: np.array([4.0]) for k in ["a", "b"]})

    def test_map_values_in_model(self, music_streaming_data):
        map_layer = mm.MapValues(tf.keras.layers.Lambda(lambda x: x + 1))

        model = mm.Model(
            mm.InputBlock(music_streaming_data.schema),
            map_layer,
            mm.MLPBlock([10]),
            mm.BinaryClassificationTask("click"),
        )

        testing_utils.model_test(model, music_streaming_data)
