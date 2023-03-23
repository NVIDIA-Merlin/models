import torch
import torch.nn as nn

from merlin.models.torch.core.base import Block, Selectable, TabularBlock
from merlin.schema import Schema


class TestBlock:
    def test_no_pre_post(self):
        block = Block()
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)

        assert torch.equal(inputs, outputs)

    def test_pre(self):
        pre = nn.Linear(2, 3)
        block = Block(pre=pre)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)
        expected_outputs = pre(inputs)

        assert torch.equal(outputs, expected_outputs)

    def test_post(self):
        post = nn.Linear(2, 3)
        block = Block(post=post)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)
        expected_outputs = post(inputs)

        assert torch.equal(outputs, expected_outputs)

    def test_pre_post(self):
        pre = nn.Linear(2, 3)
        post = nn.Linear(3, 4)
        block = Block(pre=pre, post=post)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)
        expected_outputs = pre(inputs)
        expected_outputs = post(expected_outputs)

        assert torch.equal(outputs, expected_outputs)


class TestSelectable:
    def test_schema_selectable(self):
        schema = Schema(["a", "b", "c"])

        assert isinstance(schema, Selectable)

    def test_tabular_default_selectable(self):
        block = TabularBlock()

        assert not isinstance(block, Selectable)


class TestTabularBlock:
    def test_no_pre_post_aggregation(self):
        block = TabularBlock()
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)

        assert torch.equal(inputs, outputs)

    def test_aggregation(self):
        aggregation = nn.Linear(2, 3)
        block = TabularBlock(aggregation=aggregation)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = block(inputs)
        expected_outputs = aggregation(inputs)

        assert torch.equal(outputs, expected_outputs)
