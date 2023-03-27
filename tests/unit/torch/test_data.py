import torch
from torch import nn

from merlin.models.torch.data import register_target_hook
from merlin.models.torch.models.base import Model


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
        outputs = model(torch.rand(5, 10), targets=targets)

        assert torch.allclose(outputs, targets)

    def test_target_transformation(self):
        model = Model(TargetTransformation(), TargetConsumer())

        target = torch.rand(5, 1)
        targets = {"target": target.clone()}
        outputs = model(torch.rand(5, 10), targets=targets)

        assert torch.allclose(target * 2, outputs["target"])
