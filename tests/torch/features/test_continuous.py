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

from merlin.graph.tags import Tags

import merlin_models.torch as ml


def test_continuous_features(torch_con_features):
    features = ["con_a", "con_b"]
    con = ml.ContinuousFeatures(features)(torch_con_features)

    assert list(con.keys()) == features


def test_continuous_features_yoochoose(tabular_schema, torch_tabular_data):
    cont_cols = tabular_schema.select_by_tag(Tags.CONTINUOUS)

    con = ml.ContinuousFeatures.from_schema(cont_cols)
    outputs = con(torch_tabular_data)

    assert set(outputs.keys()) == set(cont_cols.column_names)
