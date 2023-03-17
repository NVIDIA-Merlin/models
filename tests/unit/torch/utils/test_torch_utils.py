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
