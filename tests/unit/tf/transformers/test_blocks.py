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
from merlin.schema import ColumnSchema, Schema, Tags


def test_seq_random_target_masking():
    item_ids_input = tf.ragged.constant([[1, 2], [3, 4, 5], [6, 7, 8, 9]])
    schema = Schema(
        [ColumnSchema("item_id", tags=[Tags.ITEM, Tags.ITEM_ID, Tags.CATEGORICAL], dtype="int64")]
    )
    input_features = {"item_id": item_ids_input}
    emb = tf.keras.layers.Embedding(10, 5)
    inputs = emb(item_ids_input)

    masking = mm.SequenceRandomTargetMasking(schema, masking_prob=0.3)
    output = masking(inputs, features=input_features, training=True)

    masked_elements = tf.logical_not(tf.reduce_all(output.outputs == inputs, axis=-1))
    # Checks if only elements maskes as targets had the corresponding embeddings replaced
    tf.assert_equal(tf.reduce_all(masked_elements == output.outputs._keras_mask), True)
    tf.assert_equal(tf.reduce_all(masked_elements == masking.target_mask), True)

    # Checking if there is no sequence with no elements masked as target
    tf.assert_equal(tf.reduce_all(tf.reduce_any(output.outputs._keras_mask, axis=1)), True)
    # Checking if there is no sequence with all elements masked
    tf.assert_equal(tf.reduce_any(tf.reduce_all(output.outputs._keras_mask, axis=1)), False)


def test_seq_random_target_should_not_mask_if_not_training():
    item_ids_input = tf.ragged.constant([[1, 2], [3, 4, 5], [6, 7, 8, 9]])
    schema = Schema(
        [ColumnSchema("item_id", tags=[Tags.ITEM, Tags.ITEM_ID, Tags.CATEGORICAL], dtype="int64")]
    )
    input_features = {"item_id": item_ids_input}
    emb = tf.keras.layers.Embedding(10, 5)
    inputs = emb(item_ids_input)

    masking = mm.SequenceRandomTargetMasking(schema, masking_prob=0.3)
    output = masking(inputs, features=input_features, training=False)

    assert output.targets is None
    tf.assert_equal(tf.reduce_all(output.outputs == inputs), True)
    assert getattr(output.outputs, "_keras_mask", None) is None


def test_seq_random_target_masking_no_ragged_input():
    schema = Schema(
        [ColumnSchema("item_id", tags=[Tags.ITEM, Tags.ITEM_ID, Tags.CATEGORICAL], dtype="int64")]
    )
    masking = mm.SequenceRandomTargetMasking(schema)
    inputs = tf.convert_to_tensor([[[1, 2]], [[3, 4]]])
    with pytest.raises(ValueError) as exc_info:
        _ = masking(inputs)
    assert "Only a tf.RaggedTensor is accepted" in str(exc_info.value)


def test_seq_random_target_masking_no_3d_input():
    schema = Schema(
        [ColumnSchema("item_id", tags=[Tags.ITEM, Tags.ITEM_ID, Tags.CATEGORICAL], dtype="int64")]
    )
    masking = mm.SequenceRandomTargetMasking(schema)
    inputs = tf.ragged.constant([[1, 2], [3, 4, 5], [6, 7, 8, 9]])
    with pytest.raises(ValueError) as exc_info:
        _ = masking(inputs)
    assert "The inputs must be a 3D tf.RaggedTensor" in str(exc_info.value)


def test_seq_random_target_masking_input_last_dim_not_none():
    schema = Schema(
        [ColumnSchema("item_id", tags=[Tags.ITEM, Tags.ITEM_ID, Tags.CATEGORICAL], dtype="int64")]
    )
    masking = mm.SequenceRandomTargetMasking(schema)
    inputs = tf.ragged.constant([[[0.1, 0.2], [0.3, 0.4]], []])
    with pytest.raises(Exception) as exc_info:
        _ = masking(inputs)
    assert "The tf.RaggedTensor last dim cannot be None" in str(exc_info.value)


def test_seq_random_target_masking_input_seqs_dim_less_than_two():
    item_ids_input = tf.ragged.constant([[1, 2], [3, 4, 5], [6], []])
    schema = Schema(
        [ColumnSchema("item_id", tags=[Tags.ITEM, Tags.ITEM_ID, Tags.CATEGORICAL], dtype="int64")]
    )
    input_features = {"item_id": item_ids_input}
    emb = tf.keras.layers.Embedding(10, 5)
    inputs = emb(item_ids_input)

    masking = mm.SequenceRandomTargetMasking(schema)

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = masking(inputs, features=input_features, training=True)


def test_seq_random_target_masking_input_item_id_not_provided():
    item_ids_input = tf.ragged.constant([[1, 2], [3, 4, 5], [6], []])
    schema = Schema(
        [ColumnSchema("item_id", tags=[Tags.ITEM, Tags.ITEM_ID, Tags.CATEGORICAL], dtype="int64")]
    )
    emb = tf.keras.layers.Embedding(10, 5)
    inputs = emb(item_ids_input)

    masking = mm.SequenceRandomTargetMasking(schema)

    with pytest.raises(ValueError) as exc_info:
        _ = masking(inputs, features={}, training=True)
    assert "The features provided does contain the item id" in str(exc_info.value)


def test_seq_random_target_masking_serialization():
    schema = Schema(
        [ColumnSchema("item_id", tags=[Tags.ITEM, Tags.ITEM_ID, Tags.CATEGORICAL], dtype="int64")]
    )
    masking = mm.SequenceRandomTargetMasking(schema)
    config = masking.get_config()
    clone = mm.SequenceRandomTargetMasking.from_config(config)
    assert masking.schema == clone.schema
    assert masking.masking_prob == clone.masking_prob
