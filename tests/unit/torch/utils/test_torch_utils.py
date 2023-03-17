from copy import deepcopy

import torch
import torch.nn as nn

from merlin.models.torch.utils.torch_utils import apply_module, has_custom_call


class NoArgsModule(nn.Module):
    def forward(self, x):
        return x * 2


class ArgsModule(nn.Module):
    def forward(self, x, factor):
        return x * factor


class KwargsModule(nn.Module):
    def forward(self, x, factor=1, add=0):
        return x * factor + add


class CustomCallModule(nn.Module):
    def forward(self, x):
        return x * 2

    def __call__(self, x):
        return self.forward(x) + 1


class Test_apply_module:
    def test_no_args_module(self):
        module = NoArgsModule()
        x = torch.tensor([1, 2, 3])
        y = apply_module(module, x)

        assert torch.allclose(y, x * 2)

    def test_args_module(self):
        module = ArgsModule()
        x = torch.tensor([1, 2, 3])
        y = apply_module(module, x, 3)

        assert torch.allclose(y, x * 3)

    def test_kwargs_module(self):
        module = KwargsModule()
        x = torch.tensor([1, 2, 3])
        y = apply_module(module, x, factor=3, add=5)

        assert torch.allclose(y, x * 3 + 5)

    def test_custom_call_module(self):
        module = CustomCallModule()
        x = torch.tensor([1, 2, 3])
        y = apply_module(module, x)

        assert torch.allclose(y, x * 2 + 1)
        assert has_custom_call(module)

    def test_has_custom_call(self):
        no_args_module = NoArgsModule()
        args_module = ArgsModule()
        kwargs_module = KwargsModule()
        custom_call_module = CustomCallModule()

        assert not has_custom_call(no_args_module)
        assert not has_custom_call(args_module)
        assert not has_custom_call(kwargs_module)
        assert has_custom_call(custom_call_module)


class Test_deepcopy_module:
    def test_copy_module_linear(self):
        linear = nn.Linear(5, 3)
        linear_copy = deepcopy(linear)

        assert isinstance(linear_copy, nn.Linear)
        assert linear_copy.in_features == linear.in_features
        assert linear_copy.out_features == linear.out_features

        assert torch.allclose(linear.weight, linear_copy.weight)
        assert torch.allclose(linear.bias, linear_copy.bias)

        # Modify the original module weights to ensure the copied module's weights are not affected
        linear.weight.data.fill_(1.0)
        assert not torch.allclose(linear.weight, linear_copy.weight)

    def test_copy_module_conv2d(self):
        conv = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        conv_copy = deepcopy(conv)

        assert isinstance(conv_copy, nn.Conv2d)
        assert conv_copy.in_channels == conv.in_channels
        assert conv_copy.out_channels == conv.out_channels
        assert conv_copy.kernel_size == conv.kernel_size

        assert torch.allclose(conv.weight, conv_copy.weight)
        assert torch.allclose(conv.bias, conv_copy.bias)

        # Modify the original module weights to ensure the copied module's weights are not affected
        conv.weight.data.fill_(1.0)
        assert not torch.allclose(conv.weight, conv_copy.weight)

    def test_copy_module_batchnorm2d(self):
        bn = nn.BatchNorm2d(16)
        bn_copy = deepcopy(bn)

        assert isinstance(bn_copy, nn.BatchNorm2d)
        assert bn_copy.num_features == bn.num_features

        assert torch.allclose(bn.weight, bn_copy.weight)
        assert torch.allclose(bn.bias, bn_copy.bias)

        # Modify the original module weights to ensure the copied module's weights are not affected
        bn.weight.data.fill_(1.0)
        assert not torch.allclose(bn.weight, bn_copy.weight)
