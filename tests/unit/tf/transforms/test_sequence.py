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
from merlin.models.tf.loader import Loader
from merlin.models.tf.utils.testing_utils import assert_output_shape
from merlin.schema import Tags


@pytest.mark.parametrize("use_loader", [False, True])
def test_seq_predict_next(sequence_testing_data: Dataset, use_loader: bool):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.SequencePredictNext(schema=seq_schema, target=target, pre=mm.ListToRagged())

    batch = mm.sample_batch(sequence_testing_data, batch_size=8, include_targets=False)
    if use_loader:
        dataset_transformed = Loader(
            sequence_testing_data, batch_size=8, shuffle=False, transform=predict_next
        )
        output = next(iter(dataset_transformed))
    else:
        output = predict_next(batch)
    output_x, output_y = output

    as_ragged = mm.ListToRagged()
    batch = as_ragged(batch)

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


@pytest.mark.parametrize("use_loader", [False, True])
def test_seq_predict_last(sequence_testing_data: Dataset, use_loader: bool):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_last = mm.SequencePredictLast(schema=seq_schema, target=target)

    batch = mm.sample_batch(sequence_testing_data, batch_size=8, include_targets=False)
    if use_loader:
        dataset_transformed = Loader(
            sequence_testing_data, batch_size=8, shuffle=False, transform=predict_last
        )
        output = next(iter(dataset_transformed))
    else:
        output = predict_last(batch)
    output_x, output_y = output

    as_ragged = mm.ListToRagged()
    batch = as_ragged(batch)

    # Checks if sequential input features were truncated in the last position
    for k, v in batch.items():
        if k in seq_schema.column_names:
            tf.Assert(tf.reduce_all(output_x[k] == v[:, :-1]), [output_x[k], v[:, :-1]])
        else:
            tf.Assert(tf.reduce_all(output_x[k] == v), [output_x[k], v])

    expected_target = tf.squeeze(tf.sparse.to_dense(batch[target][:, -1:].to_sparse()), axis=1)
    # Checks if the target is the last item
    tf.Assert(
        tf.reduce_all(output_y == expected_target),
        [output_y, expected_target],
    )


@pytest.mark.parametrize("use_loader", [False, True])
def test_seq_predict_random(sequence_testing_data: Dataset, use_loader: bool):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_random = mm.SequencePredictRandom(schema=seq_schema, target=target)

    batch = mm.sample_batch(sequence_testing_data, batch_size=8, include_targets=False)
    if use_loader:
        dataset_transformed = Loader(
            sequence_testing_data, batch_size=8, shuffle=False, transform=predict_random
        )
        output = next(iter(dataset_transformed))
    else:
        output = predict_random(batch)
    output_x, output_y = output

    as_ragged = mm.ListToRagged()
    batch = as_ragged(batch)
    batch_size = batch[target].shape[0]

    for k, v in batch.items():

        if k in seq_schema.column_names:
            # Check if output sequences length is smaller than input sequences length
            tf.Assert(
                tf.reduce_all(output_x[k].row_lengths(1) < v.row_lengths(1)), [output_x[k], v]
            )
            # Check if first position of input and output sequence matches
            tf.Assert(
                tf.reduce_all(
                    tf.sparse.to_dense(output_x[k][:, :1].to_sparse())
                    == tf.sparse.to_dense(v[:, :1].to_sparse())
                ),
                [output_x[k], v],
            )
        else:
            tf.Assert(tf.reduce_all(output_x[k] == v), [output_x[k], v])

    # Checks if the target has the right shape
    tf.Assert(tf.reduce_all(tf.shape(output_y) == batch_size), [])


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
