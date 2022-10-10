#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

from merlin.models.tf.inputs import base
from merlin.schema import ColumnSchema, Schema, Tags


class TestInputBlockV2:
    def test_raises_with_target(self):
        schema = Schema([ColumnSchema("feature", tags=[Tags.TARGET])])
        with pytest.raises(ValueError) as exc_info:
            base.InputBlockV2(schema)
        assert "`schema` should not contain any target features" in str(exc_info.value)
