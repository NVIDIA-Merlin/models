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

        outputs = self.router(self.batch.features)
        assert set(outputs.keys()) == set(self.schema.select_by_tag(Tags.CONTINUOUS).column_names)
        assert len(self.router["continuous"]) == 2
        assert isinstance(self.router["continuous"][0], mm.SelectKeys)
        assert isinstance(self.router["continuous"][1], CustomSelect)

    def test_module_with_setup(self):
        class Dummy(nn.Module):
            def setup_schema(self, schema: Schema):
                self.schema = schema

            def forward(self, x):
                return x

        dummy = Dummy()
        self.router.add_route(Tags.CONTINUOUS, dummy)
        assert dummy.schema == mm.select_schema(self.schema, Tags.CONTINUOUS)

        dummy_2 = Dummy()
        self.router.add_route_for_each(ColumnSchema("user_id"), dummy_2, shared=True)
        assert dummy_2.schema == mm.select_schema(self.schema, ColumnSchema("user_id"))

    def test_add_route_parallel_block(self):
        class FakeEmbeddings(mm.ParallelBlock):
            ...

        self.router.add_route(Tags.CATEGORICAL, FakeEmbeddings())
        assert isinstance(self.router["categorical"], FakeEmbeddings)

    @pytest.mark.parametrize("shared", [True, False])
    def test_add_route_for_each(self, shared):
        block = mm.Block(mm.Concat(), ToFloat(), nn.LazyLinear(10)).to(self.batch.device())
        self.router.add_route_for_each(Tags.CONTINUOUS, block, shared=shared)

        dense_pos = self.router.branches["position"][1][-1]
        dense_age = self.router.branches["user_age"][1][-1]
        if shared:
            assert dense_pos == dense_age
        else:
            assert dense_pos != dense_age

        outputs = self.router(self.batch.features)
        assert set(outputs.keys()) == set(self.schema.select_by_tag(Tags.CONTINUOUS).column_names)

        for value in outputs.values():
            assert value.shape[-1] == 10

    def test_add_route_for_each_list(self):
        self.router.add_route_for_each([ColumnSchema("user_id")], ToFloat())
        assert isinstance(self.router.branches["user_id"][1], ToFloat)

    def test_select(self):
        plus_one = PlusOneDict()

        self.router.add_route(Tags.CONTINUOUS)
        self.router.add_route(Tags.CATEGORICAL)
        self.router.prepend(plus_one)

        router = self.router.select(Tags.CATEGORICAL)
        assert router.selectable.schema == self.schema.select_by_tag(Tags.CATEGORICAL)
        assert router[0][0] == plus_one

    def test_double_add(self):
        self.router.add_route(Tags.CONTINUOUS)
        with pytest.raises(ValueError):
            self.router.add_route(Tags.CONTINUOUS)

    def test_nested(self):
        self.router.add_route(Tags.CONTINUOUS)

        nested = self.router.nested_router()
        nested.add_route(Tags.USER)
        assert "user" in nested

        outputs = module_utils.module_test(nested, self.batch.features)
        assert list(outputs.keys()) == ["user_age"]
        assert "user_age" in nested.output_schema().column_names


class TestSelectKeys:
    @pytest.fixture(autouse=True)
    def setup_method(self, music_streaming_data):
        self.batch: Batch = sample_batch(music_streaming_data, batch_size=10)
        self.schema: Schema = music_streaming_data.schema
        self.user_schema: Schema = mm.select_schema(self.schema, Tags.USER)

    def test_forward(self):
        select_user = mm.SelectKeys(self.user_schema)
        outputs = select_user(self.batch.features)

        assert select_user.schema == self.user_schema

        for col in {"user_id", "country", "user_age"}:
            assert col in outputs

        assert "user_genres__values" in outputs
        assert "user_genres__offsets" in outputs

    def test_select(self):
        select_user = mm.SelectKeys(self.user_schema)

        user_id = Schema([self.user_schema["user_id"]])
        assert select_user.select(ColumnSchema("user_id")).schema == user_id
        assert select_user.select(Tags.USER).schema == self.user_schema

    def test_setup_schema(self):
        select_user = mm.SelectKeys()
        select_user.setup_schema(self.user_schema["user_id"])
        assert select_user.schema == Schema([self.user_schema["user_id"]])
