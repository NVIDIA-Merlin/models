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

from typing import Dict

import pytest
import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.block import ParallelBlock
from merlin.models.torch.schema import (
    Selectable,
    feature_schema,
    select,
    select_schema,
    selection_name,
    target_schema,
    trace,
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

    def test_select_star(self):
        output = select(self.schema, "*")
        assert output == self.schema

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

        selectable.initialize_from_schema(Schema([]))
        selectable.schema == Schema([])
        with pytest.raises(NotImplementedError):
            selectable.select(1)


class MockModule(nn.Module):
    def __init__(self, target_schema=None):
        super().__init__()
        self.target_schema = target_schema

    def forward(self, inputs, batch: Batch):
        return batch.features


class TestFeatures:
    def test_features(self):
        module = MockModule()
        features = {"a": torch.tensor([1]), "b": torch.tensor([2.3])}
        trace(module, {}, batch=Batch(features))
        assert feature_schema(module) == Schema(
            [ColumnSchema("a", dtype="int64"), ColumnSchema("b", dtype="float32")]
        )
        assert target_schema(module) == Schema()


class TestTargets:
    def test_targets(self):
        schema = Schema([ColumnSchema("a"), ColumnSchema("b")])

        module = MockModule(target_schema=schema)
        assert target_schema(module) == schema
        assert feature_schema(module) == Schema()


class TestTraceInitializeFromSchema:
    """Testing initialize_from_schema works with tracing."""

    def test_simple(self):
        class Dummy(nn.Module):
            def initialize_from_schema(self, schema: Schema):
                self.schema = schema

            def forward(self, x):
                return x

        module = Dummy()
        trace(module, {"a": torch.tensor([1])})
        assert module.schema.column_names == ["a"]

    def test_parallel_tensor(self):
        class Dummy(nn.Module):
            def initialize_from_schema(self, schema: Schema):
                self.schema = schema

            def forward(self, x: torch.Tensor):
                return x

        dummy = Dummy()
        identity = nn.Identity()
        module = ParallelBlock({"foo": dummy, "bar": identity})
        trace(module, torch.tensor([1]))
        assert dummy.schema.column_names == ["input"]

    def test_parallel_dict(self):
        class Dummy(nn.Module):
            def initialize_from_schema(self, schema: Schema):
                self.schema = schema

            def forward(self, x: Dict[str, torch.Tensor]):
                return x

        dummy = Dummy()
        module = ParallelBlock({"foo": dummy})
        trace(module, {"a": torch.tensor([1])})
        assert dummy.schema.column_names == ["a"]

    def test_sequential(self):
        class Dummy(nn.Module):
            def initialize_from_schema(self, schema: Schema):
                self.schema = schema

            def forward(self, x: Dict[str, torch.Tensor]):
                output = {}
                for k, v in x.items():
                    output[f"{k}_output"] = v
                return output

        first = Dummy()
        second = Dummy()
        module = nn.Sequential(first, second)
        trace(module, {"a": torch.tensor([1])})
        assert first.schema.column_names == ["a"]
        assert second.schema.column_names == ["a_output"]
