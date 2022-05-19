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


def test_tabular_features(testing_data: Dataset):
    tab_module = ml.InputBlock(testing_data.schema)

    outputs = tab_module(ml.sample_batch(testing_data, batch_size=100, include_targets=False))

    con = testing_data.schema.select_by_tag(Tags.CONTINUOUS).column_names
    cat = testing_data.schema.select_by_tag(Tags.CATEGORICAL).column_names

    assert set(outputs.keys()) == set(con + cat)


def test_serialization_tabular_features(testing_data: Dataset):
    inputs = ml.InputBlock(testing_data.schema)

    copy_layer = testing_utils.assert_serialization(inputs)

    assert list(inputs.parallel_layers.keys()) == list(copy_layer.parallel_layers.keys())


def test_tabular_features_with_projection(testing_data: Dataset):
    tab_module = ml.InputBlock(testing_data.schema, continuous_projection=ml.MLPBlock([64]))

    outputs = tab_module(ml.sample_batch(testing_data, batch_size=100, include_targets=False))
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
        continuous_projection = ml.MLPBlock([continuous_projection])
    inputs = ml.InputBlock(
        music_streaming_data.schema,
        continuous_projection=continuous_projection,
        aggregation="concat",
    )

    body = ml.SequentialBlock([inputs, ml.MLPBlock([64])])
    model = ml.Model(body, ml.BinaryClassificationTask("click"))

    testing_utils.model_test(model, music_streaming_data, run_eagerly=run_eagerly)
