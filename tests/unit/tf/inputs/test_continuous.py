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

import merlin.models.tf as ml
from merlin.io import Dataset
from merlin.models.tf.utils import testing_utils
from merlin.schema import Tags


def test_continuous_features(tf_con_features):
    features = ["a", "b"]
    con = ml.ContinuousFeatures(features)(tf_con_features)

    assert list(con.keys()) == features


def test_continuous_features_yoochoose(testing_data: Dataset):
    schema = testing_data.schema.select_by_tag([Tags.CONTINUOUS])

    inputs = ml.ContinuousFeatures.from_schema(schema)
    outputs = inputs(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    assert sorted(list(outputs.keys())) == sorted(schema.column_names)


def test_serialization_continuous_features(testing_data: Dataset):
    inputs = ml.ContinuousFeatures.from_schema(testing_data.schema)

    copy_layer = testing_utils.assert_serialization(inputs)

    assert inputs.filter_features.feature_names == copy_layer.filter_features.feature_names


@pytest.mark.parametrize("run_eagerly", [True, False])
def test_continuous_features_yoochoose_model(music_streaming_data: Dataset, run_eagerly):
    schema = music_streaming_data.schema.select_by_tag(Tags.CONTINUOUS)

    inputs = ml.ContinuousFeatures.from_schema(schema, aggregation="concat")
    body = ml.SequentialBlock([inputs, ml.MLPBlock([64])])
    model = ml.Model(body, ml.BinaryClassificationTask("click"))

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)
