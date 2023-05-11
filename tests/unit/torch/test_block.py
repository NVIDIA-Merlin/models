from typing import Dict

import pytest
import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.block import (
    Block,
    ContrastiveOutput,
    EmbeddingTable,
    MLPBlock,
    Model,
    ParallelBlock,
    TabularInputBlock,
)
from merlin.models.torch.utils import module_utils
from merlin.schema import Schema, Tags


class PlusOne(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + 1


class TestBlock:
    def test_identity(self):
        block = Block()

        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        outputs = module_utils.module_test(block, inputs, batch=Batch(inputs))

        assert torch.equal(inputs, outputs)

    def test_insertion(self):
        block = Block()
        block.prepend(PlusOne())
        block.append(PlusOne())

        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        outputs = module_utils.module_test(block, inputs, batch=Batch(inputs))

        assert torch.equal(outputs, inputs + 2)


class TestParallelBlock:
    def test_simple(self):
        block = ParallelBlock({"a": PlusOne(), "b": PlusOne()})

        block.rich_print()

        a = 5

    def test_insertion(self):
        plus_1 = PlusOne()
        block = ParallelBlock({"a": plus_1, "b": plus_1})
        block.append_to("a", plus_1)
        block.prepend_to("b", plus_1)
        block.append_for_each(plus_1, copy=True)

        assert len(block.branches) == 2
        assert block.branches["a"][0] == block.branches["a"][1] != block.branches["a"][2]
        assert block.branches["b"][0] == block.branches["b"][1] != block.branches["b"][2]


class TestTabularInputBlock:
    def test_mf(self, music_streaming_data):
        schema = music_streaming_data.schema
        mf_inputs = TabularInputBlock(schema)
        mf_inputs.add_for_each([Tags.USER_ID, Tags.ITEM_ID], EmbeddingTable(100))

        output = ContrastiveOutput((Tags.USER_ID, Tags.ITEM_ID), schema=schema)
        model = Model(mf_inputs, output)

        a = 5

    def test_to_tower(self, music_streaming_data):
        schema = music_streaming_data.schema
        inputs = TabularInputBlock(schema, init="defaults")
        towers = inputs.to_router()

        towers.add(Tags.USER, MLPBlock(100, 100), name="user")
        towers.add(Tags.ITEM, MLPBlock(100, 100), name="item")

        towers.rich_print()

        a = 5
