from typing import Dict

import torch
import torch.nn as nn

from merlin.models.torch.container import Parallel, WithShortcut
from merlin.models.torch.utils import module_utils


class TabularMultiply(nn.Module):
    def __init__(self, num: int, suffix: str = ""):
        super().__init__()
        self.num = num
        self.suffix = suffix

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key + self.suffix: val * self.num for key, val in inputs.items()}


class TestParallel:
    def test_single_branch(self):
        linear = nn.Linear(2, 3)
        parallel = Parallel(linear)
        x = torch.randn(4, 2)
        out = module_utils.module_test(parallel, x)

        assert isinstance(out, dict)
        assert len(out) == 1
        assert torch.allclose(out["0"], linear(x))

    def test_branch_list(self):
        layers = [nn.Linear(2, 3), nn.ReLU(), nn.Linear(2, 1)]
        parallel = Parallel(*layers)
        x = torch.randn(4, 2)
        out = module_utils.module_test(parallel, x)

        assert isinstance(out, dict)
        assert len(out) == len(layers)
        assert set(out.keys()) == {"0", "1", "2"}

        for i, layer in enumerate(layers):
            assert torch.allclose(out[str(i)], layer(x))

    def test_branch_list_dict(self):
        layers = [TabularMultiply(1, "1"), TabularMultiply(2, "2"), TabularMultiply(3, "3")]
        parallel = Parallel(*layers)
        x = {"a": torch.randn(4, 2)}
        out = module_utils.module_test(parallel, x)

        assert isinstance(out, dict)
        assert len(out) == len(layers)
        assert set(out.keys()) == {"a1", "a2", "a3"}

        for i, layer in enumerate(layers):
            assert torch.allclose(out["a" + str(i + 1)], layer(x)["a" + str(i + 1)])


class TestWithShortcut:
    def test_with_shortcut_init(self):
        linear = nn.Linear(5, 3)
        block = WithShortcut(linear)

        assert isinstance(block, Parallel)
        assert block.branches["output"].unwrap() == linear
        assert isinstance(block.branches["shortcut"].unwrap(), nn.Identity)

    def test_with_shortcut_forward(self):
        linear = nn.Linear(5, 3)
        block = WithShortcut(linear)

        input_tensor = torch.rand(5, 5)
        output = module_utils.module_test(block, input_tensor)

        assert "output" in output
        assert "shortcut" in output
        assert torch.allclose(output["output"], linear(input_tensor))
        assert torch.allclose(output["shortcut"], input_tensor)

    def test_with_shortcut_forward_dict(self):
        module = TabularMultiply(1, "1")
        block = WithShortcut(module)

        inputs = {"a": torch.randn(4, 2)}
        output = module_utils.module_test(block, inputs)

        assert "a" in output
        assert "a1" in output
        assert torch.allclose(output["a1"], module(inputs)["a1"])
        assert torch.allclose(output["a"], inputs["a"])
