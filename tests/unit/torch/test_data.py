import torch
from torch import nn

from merlin.models.torch.blocks.mlp import MLPBlock
from merlin.models.torch.data import DataPropagationModule, register_target_hook
from merlin.models.torch.models.base import Model
from merlin.models.torch.utils import module_utils


class TargetConsumer(nn.Module):
    def __init__(self):
        super().__init__()
        register_target_hook(self)

    def forward(self, inputs, targets=None):
        return targets


class TargetTransformation(nn.Module):
    def __init__(self):
        super().__init__()
        register_target_hook(self)

    def forward(self, inputs, targets=None):
        if isinstance(targets, dict):
            for key in targets:
                targets[key] = targets[key] * 2
        else:
            targets = targets * 2

        return inputs, targets


class TestTargetPropagation:
    def test_target_receiving(self):
        model = Model(TargetConsumer(), nn.Identity())
        targets = torch.rand(5, 1)
        # outputs = model(torch.rand(5, 10), targets=targets)
        outputs = module_utils.module_test(model, torch.rand(5, 10), targets=targets)

        assert torch.allclose(outputs, targets)

    def test_target_transformation(self):
        model = Model(TargetTransformation(), TargetConsumer())

        target = torch.rand(5, 1)
        targets = {"target": target.clone()}
        outputs = model(torch.rand(5, 10), targets=targets)

        assert torch.allclose(target * 2, outputs["target"])


class TestDataPropagationModule:
    def test_basic(self):
        module = DataPropagationModule(nn.Sequential(MLPBlock([10]), TargetConsumer()))

        compiled = torch.jit.script(module)

        assert compiled is not None
