from typing import Tuple

import pytest
import torch
from torch import nn

import merlin.models.torch as mm
from merlin.models.torch.blocks.cross import CrossBlock, LazyMirrorLinear
from merlin.models.torch.utils import module_utils


class TestLazyMirrorLinear:
    def test_init(self):
        module = LazyMirrorLinear(bias=True)
        assert isinstance(module.weight, nn.parameter.UninitializedParameter)
        assert isinstance(module.bias, nn.parameter.UninitializedParameter)

    def test_no_bias_init(self):
        module = LazyMirrorLinear(bias=False)
        assert isinstance(module.weight, nn.parameter.UninitializedParameter)
        assert module.bias is None

    def test_reset_parameters(self):
        module = LazyMirrorLinear(bias=True)
        input = torch.randn(10, 20)
        module.initialize_parameters(input)
        assert module.in_features == 20
        assert module.out_features == 20
        assert module.weight.shape == (20, 20)
        assert module.bias.shape == (20,)

    def test_forward(self):
        module = LazyMirrorLinear(bias=True)
        input = torch.randn(10, 20)
        output = module_utils.module_test(module, input)
        assert output.shape == (10, 20)

    def test_no_bias_forward(self):
        module = LazyMirrorLinear(bias=False)
        input = torch.randn(10, 20)
        output = module_utils.module_test(module, input)
        assert output.shape == (10, 20)


class TestCrossBlock:
    def test_with_depth(self):
        crossblock = CrossBlock.with_depth(depth=1)
        assert len(crossblock) == 1
        assert isinstance(crossblock[0][0], LazyMirrorLinear)

    def test_with_multiple_depth(self):
        crossblock = CrossBlock.with_depth(depth=3)
        assert len(crossblock) == 3
        for module in crossblock:
            assert isinstance(module[0], LazyMirrorLinear)

    def test_crossblock_invalid_depth(self):
        with pytest.raises(ValueError):
            CrossBlock.with_depth(depth=0)

    def test_forward_tensor(self):
        crossblock = CrossBlock.with_depth(depth=1)
        input = torch.randn(5, 10)
        output = module_utils.module_test(crossblock, input)
        assert output.shape == (5, 10)

    def test_forward_dict(self):
        crossblock = CrossBlock.with_depth(depth=1)
        inputs = {"a": torch.randn(5, 10), "b": torch.randn(5, 10)}
        output = module_utils.module_test(crossblock, inputs)
        assert output.shape == (5, 20)

    def test_forward_multiple_depth(self):
        crossblock = CrossBlock.with_depth(depth=3)
        input = torch.randn(5, 10)
        output = module_utils.module_test(crossblock, input)
        assert output.shape == (5, 10)

    def test_with_low_rank(self):
        crossblock = CrossBlock.with_low_rank(depth=2, low_rank=mm.MLPBlock([5]))
        assert len(crossblock) == 2

        input = torch.randn(5, 10)
        output = module_utils.module_test(crossblock, input)
        assert output.shape == (5, 10)

        assert crossblock[0][0][1].in_features == 10
        assert crossblock[0][0][1].out_features == 5
        assert crossblock[0][1].in_features == 5
        assert crossblock[0][1].out_features == 10

    def test_exception(self):
        class ToTuple(nn.Module):
            def forward(self, input) -> Tuple[torch.Tensor, torch.Tensor]:
                return input, input

        crossblock = CrossBlock(ToTuple())

        with pytest.raises(RuntimeError):
            module_utils.module_test(crossblock, torch.randn(5, 10))
