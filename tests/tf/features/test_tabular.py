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

from merlin_models.data.synthetic import SyntheticData
from merlin_standard_lib import Tag

ml = pytest.importorskip("merlin_models.tf")
test_utils = pytest.importorskip("merlin_models.tf.utils.testing_utils")


def test_tabular_features(testing_data: SyntheticData):
    tab_module = ml.TabularFeatures.from_schema(testing_data.schema)

    outputs = tab_module(testing_data.tf_tensor_dict)

    con = testing_data.schema.select_by_tag(Tag.CONTINUOUS).column_names
    cat = testing_data.schema.select_by_tag(Tag.CATEGORICAL).column_names

    assert set(outputs.keys()) == set(con + cat)


def test_serialization_tabular_features(testing_data: SyntheticData):
    inputs = ml.TabularFeatures.from_schema(testing_data.schema)

    copy_layer = test_utils.assert_serialization(inputs)

    assert list(inputs.parallel_layers.keys()) == list(copy_layer.parallel_layers.keys())


def test_tabular_features_with_projection(testing_data: SyntheticData):
    tab_module = ml.TabularFeatures.from_schema(testing_data.schema, continuous_projection=64)

    outputs = tab_module(testing_data.tf_tensor_dict)
    continuous_feature_names = testing_data.schema.select_by_tag(Tag.CONTINUOUS).column_names

    assert len(set(continuous_feature_names).intersection(set(outputs.keys()))) == 0
    assert "continuous_projection" in outputs
    assert list(outputs["continuous_projection"].shape)[1] == 64


@test_utils.mark_run_eagerly_modes
@pytest.mark.parametrize("continuous_projection", [None, 128])
def test_tabular_features_yoochoose_model(
    testing_data: SyntheticData, run_eagerly, continuous_projection
):
    inputs = ml.TabularFeatures.from_schema(
        testing_data.schema, continuous_projection=continuous_projection, aggregation="concat"
    )

    body = ml.SequentialBlock([inputs, ml.MLPBlock([64])])

    test_utils.assert_body_works_in_model(testing_data.tf_tensor_dict, body, run_eagerly)
