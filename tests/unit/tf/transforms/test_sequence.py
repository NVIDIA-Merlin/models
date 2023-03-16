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
from merlin.models.tf.utils.testing_utils import assert_output_shape
from merlin.schema import Tags


def test_seq_predict_next(sequence_testing_data: Dataset):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.SequencePredictNext(schema=seq_schema, target=target)

    batch, _ = mm.sample_batch(sequence_testing_data, batch_size=8, prepare_features=True)
    output = predict_next(batch)
    output_x, output_y = output
    output_y = output_y[target]

    # Checks if sequential input features were truncated in the last position
    for k, v in batch.items():
        if k in seq_schema.column_names:
            tf.Assert(tf.reduce_all(output_x[k] == v[:, :-1]), [output_x[k], v[:, :-1]])
        else:
            tf.Assert(tf.reduce_all(output_x[k] == v), [output_x[k], v])

    # Checks if the target is the shifted input feature
    tf.Assert(
        tf.reduce_all(output_y == batch[target][:, 1:]),
        [output_y, batch[target][:, 1:]],
    )


def test_seq_predict_last(sequence_testing_data: Dataset):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_last = mm.SequencePredictLast(schema=seq_schema, target=target)

    batch, _ = mm.sample_batch(sequence_testing_data, batch_size=8, prepare_features=True)
    output = predict_last(batch)
    output_x, output_y = output
    output_y = output_y[target]

    # Checks if sequential input features were truncated in the last position
    for k, v in batch.items():
        if k in seq_schema.column_names:
            tf.Assert(tf.reduce_all(output_x[k] == v[:, :-1]), [output_x[k], v[:, :-1]])
        else:
            tf.Assert(tf.reduce_all(output_x[k] == v), [output_x[k], v])

    expected_target = tf.squeeze(batch[target][:, -1:].to_tensor(), axis=1)
    # Checks if the target is the last item
    tf.Assert(
        tf.reduce_all(output_y == expected_target),
        [output_y, expected_target],
    )


def test_seq_predict_random(sequence_testing_data: Dataset):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_random = mm.SequencePredictRandom(schema=seq_schema, target=target)

    batch, _ = mm.sample_batch(sequence_testing_data, batch_size=8, prepare_features=True)
    output = predict_random(batch)
    output_x, output_y = output
    output_y = output_y[target]

    batch_size = batch[target].shape[0]

    for k, v in batch.items():
        if k in seq_schema.column_names:
            # Check if output sequences length is smaller than input sequences length
            tf.Assert(
                tf.reduce_all(output_x[k].row_lengths(1) < v.row_lengths(1)), [output_x[k], v]
            )
            # Check if first position of input and output sequence matches
            tf.Assert(
                tf.reduce_all(output_x[k][:, :1].to_tensor() == v[:, :1].to_tensor()),
                [output_x[k], v],
            )
        else:
            tf.Assert(tf.reduce_all(output_x[k] == v), [output_x[k], v])

    # Checks if the target has the right shape
    tf.Assert(tf.reduce_all(tf.shape(output_y) == tf.TensorShape((batch_size, 1))), [])


def test_seq_predict_next_output_shape(sequence_testing_data):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.SequencePredictNext(schema=seq_schema, target=target)

    batch = mm.sample_batch(sequence_testing_data, batch_size=8, include_targets=False)

    input_shapes = dict()
    expected_output_shapes = dict()
    for k, v in batch.items():
        if k in seq_schema.column_names:
            input_shapes[k] = (v[0].shape, v[1].shape)
            expected_output_shapes[k] = tf.TensorShape([v[1].shape[0], None])
        else:
            input_shapes[k] = expected_output_shapes[k] = v.shape

    # Hacking one input feature to have defined seq length
    input_shapes["categories"] = tf.TensorShape([8, 4])
    expected_output_shapes["categories"] = tf.TensorShape([8, 3])

    output_shape = predict_next.compute_output_shape(input_shapes)
    assert_output_shape(output_shape, expected_output_shapes)


def test_seq_predict_next_serialize_deserialize(sequence_testing_data):
    predict_next = mm.SequencePredictNext(sequence_testing_data.schema, "item_id_seq")
    assert isinstance(predict_next.from_config(predict_next.get_config()), mm.SequencePredictNext)


def asserts_mlm_target_mask(target_mask):
    # Checks if there is no sequence with no elements masked as target
    tf.assert_equal(
        tf.reduce_all(tf.reduce_any(target_mask, axis=1)),
        True,
        message=f"There are sequences with no targets masked {target_mask.numpy()}",
    )
    # Checks if there is no sequence with all elements masked
    tf.assert_equal(
        tf.reduce_any(tf.reduce_all(target_mask, axis=1)),
        False,
        message=f"There are sequences with all targets masked {target_mask.numpy()}",
    )


def test_seq_random_masking(sequence_testing_data: Dataset):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_masked = mm.SequenceMaskRandom(schema=seq_schema, target=target, masking_prob=0.3)

    batch, _ = mm.sample_batch(sequence_testing_data, batch_size=8, prepare_features=True)

    output_x, output_y = predict_masked(batch)
    output_y = output_y[target]

    tf.Assert(tf.reduce_all(output_y == output_x[target]), [output_y, output_x[target]])

    target_mask = output_y._keras_mask

    asserts_mlm_target_mask(target_mask)

    for k, v in batch.items():
        # Checking if inputs values didn't change
        tf.Assert(tf.reduce_all(output_x[k] == v), [output_x[k], v])

        # Checks if for sequential input columns the mask has been assigned
        # (opposite of the target mask)
        if k in seq_schema.column_names:
            tf.Assert(
                tf.reduce_all(output_x[k]._keras_mask == target_mask),
                [],
            )


def test_seq_predict_masked_target_not_present(sequence_testing_data: Dataset):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    with pytest.raises(ValueError) as exc_info:
        _ = mm.SequenceMaskRandom(schema=seq_schema, target="NOT_EXISTS", masking_prob=0.3)
    assert "The target column needs to be part of the sequential schema" in str(exc_info.value)


def test_seq_predict_masked_serialize_deserialize(sequence_testing_data):
    predict_masked = mm.SequenceMaskRandom(sequence_testing_data.schema, "item_id_seq")
    assert isinstance(
        predict_masked.from_config(predict_masked.get_config()), mm.SequenceMaskRandom
    )


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("target_as_dict", [False, True])
def test_seq_mask_random_replace_embeddings(
    sequence_testing_data: Dataset, dense: bool, target_as_dict: bool
):
    from merlin.models.tf.transforms.sequence import ExtractMaskFromTargets

    sequence_testing_data.schema = sequence_testing_data.schema.select_by_tag(
        Tags.SEQUENCE
    ).select_by_name(["item_id_seq", "categories"])
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_masked = mm.SequenceMaskRandom(
        schema=sequence_testing_data.schema, target=target, masking_prob=0.3
    )

    batch, _ = mm.sample_batch(sequence_testing_data, batch_size=8, prepare_features=False)

    inputs, targets = predict_masked(batch)
    targets = targets[target]

    emb = tf.keras.layers.Embedding(1000, 16)
    item_id_emb_seq = emb(tf.squeeze(inputs["item_id_seq"], axis=-1))
    targets_mask = targets._keras_mask
    if dense:
        item_id_emb_seq = item_id_emb_seq.to_tensor()
        targets = targets.to_tensor()
        targets_mask = targets_mask.to_tensor()
        targets._keras_mask = targets_mask

    if target_as_dict:
        # Making targets different in dict, the valid one is "target2" which is 2D
        targets = {"target1": tf.ragged.constant([1, 2, 3, 4, 5, 6, 7, 8]), "target2": targets}

    masked_embeddings = mm.SequentialBlock(ExtractMaskFromTargets(), mm.ReplaceMaskedEmbeddings())
    output = masked_embeddings(item_id_emb_seq, targets=targets, training=True)

    replaced_mask = tf.logical_not(tf.reduce_all(output == item_id_emb_seq, axis=2))

    if not dense:
        replaced_mask = replaced_mask.with_row_splits_dtype(tf.int64)
    tf.Assert(tf.reduce_all(replaced_mask == targets_mask), [])
    asserts_mlm_target_mask(replaced_mask)


def test_replace_masked_input_embeddings_no_target():
    from merlin.models.tf.transforms.sequence import ExtractMaskFromTargets

    item_id_emb_seq = tf.random.uniform((8, 10), dtype=tf.float32)
    item_id_emb_seq._keras_mask = tf.cast(
        tf.random.uniform((8,), minval=0, maxval=2, dtype=tf.int32), tf.bool
    )
    targets = None

    masked_embeddings = mm.SequentialBlock(ExtractMaskFromTargets(), mm.ReplaceMaskedEmbeddings())
    output = masked_embeddings(item_id_emb_seq, targets=targets, training=True)
    # Checks that no input embedding was replaced, as there was no masking defined
    tf.Assert(tf.logical_not(tf.reduce_all(output == item_id_emb_seq)), [])


def test_not_replace_unmasked_sequence_embeddings():
    from merlin.models.tf.transforms.sequence import ExtractMaskFromTargets

    item_id_emb_seq = tf.random.uniform((8, 10), dtype=tf.float32)
    targets = tf.random.uniform((8, 10), dtype=tf.float32)

    masked_embeddings = mm.SequentialBlock(ExtractMaskFromTargets(), mm.ReplaceMaskedEmbeddings())
    output = masked_embeddings(item_id_emb_seq, targets=targets, training=True)
    # Checks that no input embedding was replaced, as there was no masking defined
    tf.Assert(tf.reduce_all(output == item_id_emb_seq), [])


def test_replace_masked_input_2d_embeddings_incompatible_2d_mask():
    item_id_emb_seq = tf.random.uniform((8, 10), dtype=tf.float32)
    item_id_emb_seq._keras_mask = tf.cast(
        tf.random.uniform((8, 10), minval=0, maxval=2, dtype=tf.int32), tf.bool
    )
    masked_embeddings = mm.ReplaceMaskedEmbeddings()

    with pytest.raises(Exception) as exc_info:
        _ = masked_embeddings(item_id_emb_seq, training=True)
    assert "The inputs and mask need to be compatible" in str(exc_info.value)


def test_replace_masked_input_2d_embeddings_incompatible_ragged_2d_mask():
    item_id_emb_seq = tf.random.uniform((8, 10), dtype=tf.float32)
    item_id_emb_seq._keras_mask = tf.RaggedTensor.from_tensor(
        tf.cast(tf.random.uniform((8, 10), minval=0, maxval=2, dtype=tf.int32), tf.bool)
    )
    masked_embeddings = mm.ReplaceMaskedEmbeddings()

    with pytest.raises(Exception) as exc_info:
        _ = masked_embeddings(item_id_emb_seq, training=True)
    assert "The inputs and mask need to be compatible" in str(exc_info.value)


def test_seq_replace_embeddings_ragged_tensor_last_dim_none():
    inputs = tf.ragged.constant([[[1, 2]], [[3]]])

    masked_embeddings = mm.ReplaceMaskedEmbeddings()
    with pytest.raises(Exception) as exc_info:
        _ = masked_embeddings(inputs, training=True)
    assert "The last dim of inputs cannot be None" in str(exc_info.value)
