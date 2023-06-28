import pytest
import torch
from torch import nn

from merlin.models.torch.blocks.cross import CrossBlock, CrossLink, LazyMirrorLinear
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


class TestCrossLink:
    def test_exception(self):
        with pytest.raises(TypeError):
            CrossLink(0)

        with pytest.raises(TypeError):
            CrossLink(nn.Linear(10, 10))


class TestCrossBlock:
    def test_init(self):
        crossblock = CrossBlock(depth=1)
        assert len(crossblock) == 2
        assert len(crossblock[1].output) == 1

    def test_init_multiple_depth(self):
        crossblock = CrossBlock(depth=3)
        assert len(crossblock) == 2
        assert len(crossblock[1].output) == 3

    def test_crossblock_invalid_depth(self):
        with pytest.raises(ValueError):
            CrossBlock(depth=0)

    def test_forward_tensor(self):
        crossblock = CrossBlock(depth=1)
        input = torch.randn(5, 10)
        output = module_utils.module_test(crossblock, input)
        assert output.shape == (5, 10)

    def test_forward_dict(self):
        crossblock = CrossBlock(depth=1)
        inputs = {"a": torch.randn(5, 10), "b": torch.randn(5, 10)}
        output = module_utils.module_test(crossblock, inputs)
        assert output.shape == (5, 20)

    def test_forward_multiple_depth(self):
        crossblock = CrossBlock(depth=3)
        input = torch.randn(5, 10)
        output = module_utils.module_test(crossblock, input)
        assert output.shape == (5, 10)
