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

import merlin.models.torch as mm
from merlin.models.torch.batch import Batch, sample_batch
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


class ToFloat(nn.Module):
    def forward(self, x):
        return x.float()


class PlusOneDict(nn.Module):
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v + 1 for k, v in inputs.items()}

    def select(self, selection):
        return self


class TestRouterBlock:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.schema = music_streaming_data.schema
        self.router: mm.RouterBlock = mm.RouterBlock(self.schema)
        self.batch: Batch = sample_batch(music_streaming_data, batch_size=10)

    def test_add_route(self):
        self.router.add_route(Tags.CONTINUOUS)

        outputs = module_utils.module_test(self.router, self.batch.features)
        assert set(outputs.keys()) == set(self.schema.select_by_tag(Tags.CONTINUOUS).column_names)
        assert "continuous" in self.router
        assert len(self.router["continuous"]) == 1
        assert isinstance(self.router["continuous"][0], mm.SelectKeys)

    def test_add_route_module(self):
        class CustomSelect(mm.SelectKeys):
            ...

        self.router.add_route(Tags.CONTINUOUS, CustomSelect())

        outputs = module_utils.module_test(self.router, self.batch.features)
        assert set(outputs.keys()) == set(self.schema.select_by_tag(Tags.CONTINUOUS).column_names)
        assert len(self.router["continuous"]) == 2
        assert isinstance(self.router["continuous"][0], mm.SelectKeys)
        assert isinstance(self.router["continuous"][1], CustomSelect)

    def test_module_with_setup(self):
        class Dummy(nn.Module):
            def initialize_from_schema(self, schema: Schema):
                self.schema = schema

            def forward(self, x):
                return x

        dummy = Dummy()
        self.router.add_route(Tags.CONTINUOUS, dummy)
        assert dummy.schema == mm.schema.select(self.schema, Tags.CONTINUOUS)

        dummy_2 = Dummy()
        self.router.add_route_for_each(ColumnSchema("user_id"), dummy_2, shared=True)
        assert dummy_2.schema == mm.schema.select(self.schema, ColumnSchema("user_id"))

    def test_add_route_parallel_block(self):
        class FakeEmbeddings(mm.ParallelBlock):
            ...

        self.router.add_route(Tags.CATEGORICAL, FakeEmbeddings())
        assert isinstance(self.router["categorical"], FakeEmbeddings)

    @pytest.mark.parametrize("shared", [True, False])
    def test_add_route_for_each(self, shared):
        block = mm.Block(mm.Concat(), ToFloat(), nn.LazyLinear(10))
        block.to(self.batch.device())
        self.router.add_route_for_each(Tags.CONTINUOUS, block, shared=shared)

        dense_pos = self.router.branches["position"][1][-1]
        dense_age = self.router.branches["user_age"][1][-1]
        if shared:
            assert dense_pos == dense_age
        else:
            assert dense_pos != dense_age

        outputs = module_utils.module_test(self.router, self.batch.features)

        assert set(outputs.keys()) == set(self.schema.select_by_tag(Tags.CONTINUOUS).column_names)

        for value in outputs.values():
            assert value.shape[-1] == 10

    def test_add_route_for_each_list(self):
        self.router.add_route_for_each([ColumnSchema("user_id")], ToFloat())
        assert isinstance(self.router.branches["user_id"][1], ToFloat)

    def test_select(self):
        plus_one = PlusOneDict()

        self.router.add_route(Tags.CONTINUOUS)
        self.router.add_route(Tags.USER, mm.MLPBlock([10]))
        self.router.add_route(Tags.ITEM, mm.ParallelBlock({"nested": mm.MLPBlock([10])}))
        self.router.prepend(plus_one)

        user = mm.schema.select(self.router, Tags.USER)
        assert "item_recency" not in user.branches["continuous"][0].column_names
        assert "item" not in user.branches
        assert user.pre[0] == plus_one

        item = mm.schema.select(self.router, Tags.ITEM)
        assert item.branches["continuous"][0].column_names == ["item_recency"]
        assert list(item.branches["item"].branches.keys()) == ["nested"]
        assert all(c.startswith("item_") for c in item.branches["item"][0][0].column_names)

        self.router.add_route(Tags.CATEGORICAL, mm.MLPBlock([10]))

    def test_select_post(self):
        self.router.add_route(Tags.USER, mm.MLPBlock([10]))
        self.router.add_route(Tags.ITEM, mm.MLPBlock([10]))
        self.router.append(mm.Concat())

        user = mm.schema.select(self.router, Tags.USER)
        assert not user.post
        assert list(user.branches.keys()) == ["user"]

        item = mm.schema.select(self.router, Tags.ITEM)
        assert not item.post
        assert list(item.branches.keys()) == ["item"]

    def test_double_add(self):
        self.router.add_route(Tags.CONTINUOUS)
        with pytest.raises(ValueError):
            self.router.add_route(Tags.CONTINUOUS)

    def test_nested(self):
        self.router.add_route(Tags.CONTINUOUS)
        self.router(self.batch.features)

        nested = self.router.reroute()
        nested.add_route(Tags.USER)
        assert "user" in nested

        outputs = module_utils.module_test(nested, self.batch.features)
        assert list(outputs.keys()) == ["user_age"]
        assert "user_age" in mm.output_schema(nested).column_names

    def test_exceptions(self):
        router = mm.RouterBlock(None)
        with pytest.raises(ValueError):
            router.add_route(Tags.CONTINUOUS)

        router = mm.RouterBlock(self.schema, prepend_routing_module=False)
        with pytest.raises(ValueError):
            router.add_route(Tags.CONTINUOUS)
