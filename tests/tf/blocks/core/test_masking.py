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

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.schema import Tags


@pytest.mark.parametrize("mask_block", [ml.CausalLanguageModeling, ml.MaskedLanguageModeling])
def test_masking_block(sequence_testing_data: Dataset, mask_block):

    schema_list = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    embedding_block = ml.InputBlock(schema_list, aggregation="concat", max_seq_length=4, seq=True)
    model = embedding_block.connect(mask_block(), context=ml.BlockContext())

    batch = ml.sample_batch(sequence_testing_data, batch_size=100, include_targets=False)
    masked_input = model(batch)
    assert masked_input.shape[-1] == 148
    assert masked_input.shape[1] == 4


def test_masking_schema_error(sequence_testing_data: Dataset):
    schema_list = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    embedding_block = ml.InputBlock(schema_list, aggregation="concat", seq=True)
    model = embedding_block.connect(ml.MLPBlock([64]), context=ml.BlockContext())

    with pytest.raises(ValueError) as excinfo:
        _ = model.context.get_mask()
    assert "The mask schema is not stored" in str(excinfo.value)


# Test only last item is masked when eval_on_last_item_seq_only
@pytest.mark.parametrize("mask_block", [ml.CausalLanguageModeling, ml.MaskedLanguageModeling])
def test_masking_only_last_item_for_eval(sequence_testing_data, mask_block):
    schema_list = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    embedding_block = ml.InputBlock(schema_list, aggregation="concat", max_seq_length=4, seq=True)
    model = embedding_block.connect(mask_block(), context=ml.BlockContext())

    batch = ml.sample_batch(
        sequence_testing_data, batch_size=100, include_targets=False, to_dense=True
    )
    _ = model(batch, training=False)

    # Get last non-padded label from input
    item_ids = batch["item_id_seq"]
    non_padded_mask = item_ids != 0
    rows_ids = tf.range(item_ids.shape[0], dtype=tf.int64)
    last_item_sessions = tf.reduce_sum(tf.cast(non_padded_mask, tf.int64), axis=1) - 1
    indices = tf.concat(
        [tf.expand_dims(rows_ids, 1), tf.expand_dims(last_item_sessions, 1)], axis=1
    )
    last_labels = tf.gather_nd(item_ids, indices).numpy()
    # get the mask schema from the model
    trgt_mask = model.context.get_mask()

    # check that only one item is masked for each session
    assert tf.reduce_sum(tf.cast(trgt_mask, tf.int32)).numpy() == batch["item_id_seq"].shape[0]
    # check only the last non-paded item is masked
    out_targets = tf.boolean_mask(batch["item_id_seq"], trgt_mask).numpy()
    assert all(last_labels == out_targets)


# Test at least one item is masked when training
def test_at_least_one_masked_item_mlm(sequence_testing_data):
    schema_list = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    embedding_block = ml.InputBlock(schema_list, max_seq_length=4, aggregation="concat", seq=True)
    mask_block = ml.MaskedLanguageModeling()
    model = embedding_block.connect(mask_block, context=ml.BlockContext())

    batch = ml.sample_batch(sequence_testing_data, batch_size=100, include_targets=False)
    _ = model(batch, training=True)

    trgt_mask = tf.cast(model.context.get_mask(), tf.int32)
    assert all(tf.reduce_sum(trgt_mask, axis=1).numpy() > 0)


# Check that not all items are masked when training
def test_not_all_masked_lm(sequence_testing_data):
    schema_list = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    embedding_block = ml.InputBlock(schema_list, aggregation="concat", seq=True, max_seq_length=4)
    mask_block = ml.MaskedLanguageModeling()
    model = embedding_block.connect(mask_block, context=ml.BlockContext())

    batch = ml.sample_batch(
        sequence_testing_data, batch_size=100, include_targets=False, to_dense=True
    )
    _ = model(batch, training=True)

    trgt_mask = tf.cast(model.context.get_mask(), tf.int32)
    non_padded_mask = batch["item_id_seq"] != 0
    assert all(trgt_mask.numpy().sum(axis=1) != non_padded_mask.numpy().sum(axis=1))
