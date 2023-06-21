#
# Copyright (c) 2023, NVIDIA CORPORATION.
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
from torch import nn

from merlin.models.torch.schema import (
    Selectable,
    features,
    select,
    select_schema,
    selection_name,
    targets,
)
from merlin.schema import ColumnSchema, Schema, Tags


class TestSelectSchema:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.schema: Schema = music_streaming_data.schema

    def test_select_schema(self):
        selection = self.schema.select_by_tag(Tags.USER)
        output = select(self.schema, selection)

        assert output == selection

    def test_select_tag(self):
        selection = self.schema.select_by_tag(Tags.USER)
        output = select(self.schema, Tags.USER)

        assert output == selection

    def test_select_callable(self):
        def selection_callable(schema: Schema):
            return schema.select_by_tag(Tags.USER)

        selection = self.schema.select_by_tag(Tags.USER)
        output = select(self.schema, selection_callable)
        assert output == selection

    def test_select_column(self):
        column = self.schema["user_id"]

        output = select(self.schema, column)
        output_2 = select(self.schema, ColumnSchema("user_id"))
        assert output == output_2 == Schema([column])

    def test_exceptions(self):
        with pytest.raises(ValueError, match="is not valid"):
            select(self.schema, 1)

        with pytest.raises(ValueError, match="is not valid"):
            select_schema(1, 1)


class TestSelectionName:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.schema: Schema = music_streaming_data.schema

    def test_select_schema(self):
        selection = self.schema.select_by_tag(Tags.USER)

        assert selection_name(selection) == "_".join(selection.column_names)

    def test_select_tag(self):
        selection = Tags.USER

        assert selection_name(selection) == selection.value

    def test_select_callable(self):
        def selection_callable(schema: Schema):
            return schema.select_by_tag(Tags.USER)

        assert selection_name(selection_callable) == selection_callable.__name__

    def test_select_column(self):
        column = self.schema["user_id"]

        assert selection_name(column) == column.name
        assert selection_name(ColumnSchema("user_id")) == column.name

    def test_exception(self):
        with pytest.raises(ValueError, match="is not valid"):
            selection_name(1)


class TestSelectable:
    def test_exception(self):
        selectable = Selectable()

        selectable.setup_schema(Schema([]))
        selectable.schema == Schema([])
        with pytest.raises(NotImplementedError):
            selectable.select(1)


class MockModule(nn.Module):
    def __init__(self, feature_schema=None, target_schema=None):
        super().__init__()
        self.feature_schema = feature_schema
        self.target_schema = target_schema


class TestFeatures:
    def test_features(self):
        schema = Schema([ColumnSchema("a"), ColumnSchema("b")])

        module = MockModule(feature_schema=schema)
        assert features(module) == schema
        assert targets(module) == Schema()


class TestTargets:
    def test_targets(self):
        schema = Schema([ColumnSchema("a"), ColumnSchema("b")])

        module = MockModule(target_schema=schema)
        assert targets(module) == schema
        assert features(module) == Schema()
