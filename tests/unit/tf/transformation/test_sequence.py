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

import tensorflow as tf

import merlin.models.tf as mm
from merlin.io import Dataset
from merlin.models.tf.dataset import BatchedDataset
from merlin.models.tf.utils.testing_utils import assert_output_shape
from merlin.schema import Tags


def test_predict_next(sequence_testing_data: Dataset):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.PredictNext(schema=seq_schema, target=target)

    batch = mm.sample_batch(sequence_testing_data, batch_size=8, include_targets=False)
    output = predict_next(batch)

    as_ragged = mm.AsRaggedFeatures()
    batch = as_ragged(batch)

    # Checks if sequential input features were truncated in the last position
    for k, v in batch.items():
        if k in seq_schema.column_names:
            tf.Assert(tf.reduce_all(output.outputs[k] == v[:, :-1]), [output.outputs[k], v[:, :-1]])
        else:
            tf.Assert(tf.reduce_all(output.outputs[k] == v), [output.outputs[k], v])

    # Checks if the target is the shifted input feature
    tf.Assert(
        tf.reduce_all(output.targets == batch[target][:, 1:]),
        [output.targets, batch[target][:, 1:]],
    )


def test_predict_next_with_loader(sequence_testing_data: Dataset):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.PredictNext(schema=seq_schema, target=target)

    batch = mm.sample_batch(
        sequence_testing_data, batch_size=8, to_ragged=True, include_targets=False
    )

    dataset_transformed = BatchedDataset(sequence_testing_data, batch_size=8, shuffle=False).map(
        predict_next
    )
    batch_transformed = next(iter(dataset_transformed))

    for col_name, col_val in batch.items():
        if col_name in seq_schema.column_names:
            tf.Assert(
                tf.reduce_all(batch_transformed.outputs[col_name] == batch[col_name][:, :-1]), []
            )
        else:
            tf.Assert(tf.reduce_all(batch_transformed.outputs[col_name] == batch[col_name]), [])

    tf.Assert(tf.reduce_all(batch_transformed.targets == batch[target][:, 1:]), [])


def test_predict_next_output_shape(sequence_testing_data):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.PredictNext(schema=seq_schema, target=target)

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


def test_predict_next_output_schema(sequence_testing_data):
    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    target = sequence_testing_data.schema.select_by_tag(Tags.ITEM_ID).column_names[0]
    predict_next = mm.PredictNext(schema=seq_schema, target=target)

    output_schema = predict_next.compute_output_schema(sequence_testing_data.schema)

    for col_name, col_schema in sequence_testing_data.schema.column_schemas.items():
        output_col = output_schema.select_by_name(col_name).first
        if col_name in seq_schema.column_names:
            assert output_col.value_count.max == col_schema.value_count.max - 1
        else:
            assert output_col == col_schema
