from typing import Optional

import torch
from torch import nn

from merlin.models.torch.data import sample_batch
from merlin.models.torch.testing import (
    Block,
    BlockMixin,
    TabularBatch,
    TabularPadding,
    TabularSequence,
)
from merlin.models.torch.utils import module_utils
from merlin.schema import Tags


class WithBatch(nn.Module):
    def forward(self, x, batch: Optional[TabularBatch]):
        return x * 2


class WithoutBatch(nn.Module):
    def forward(self, x):
        return x * 3


class TestBlockMixin:
    def test_register_block_hooks(self):
        pre_module = WithBatch()
        post_module = WithoutBatch()
        block = Block(nn.Identity(), pre=pre_module, post=post_module)

        assert block.pre.to_call == pre_module
        assert block.post.to_call == post_module
        assert block.pre.needs_batch is True
        assert block.post.needs_batch is False

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output = module_utils.module_test(block, input_tensor)

        assert torch.allclose(output, input_tensor * 3 * 2)

    def test_sequential_module(self):
        block = Block(nn.Sequential(WithoutBatch(), WithBatch()))

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output = module_utils.module_test(block, input_tensor)
        assert torch.allclose(output, input_tensor * 3 * 2)

    def test_block_mixin_call(self):
        class ExampleModule(nn.Module):
            def forward(self, x):
                return x * 2

        pre_module = ExampleModule()
        post_module = ExampleModule()

        block_mixin = BlockMixin()
        block_mixin.register_block_hooks(pre=pre_module, post=post_module)

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output = block_mixin.__call__(input_tensor)

        assert torch.allclose(output, input_tensor * 4)


def test_padding(sequence_testing_data):
    features, targets = sample_batch(sequence_testing_data, 5)

    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    padding = TabularPadding(seq_schema, seq_schema)

    batch = TabularBatch(features, targets)
    out = module_utils.module_test(padding, batch)
    # out = padding(TabularBatch(features, targets))

    assert isinstance(out, TabularBatch)
    assert isinstance(out.sequences, TabularSequence)
