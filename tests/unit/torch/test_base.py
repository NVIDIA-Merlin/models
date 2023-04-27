from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional

from merlin.models.torch.base import Block, TabularBlock
from merlin.models.torch.utils import module_utils
from merlin.schema import Schema


class ConcatDict(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(x.values()), dim=-1)


class TestBlock:
    def test_no_pre_post(self):
        block = Block()
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = module_utils.module_test(block, inputs)

        assert torch.equal(inputs, outputs)

    def test_pre(self):
        pre = nn.Linear(2, 3)
        block = Block(pre=pre)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = module_utils.module_test(block, inputs)
        expected_outputs = pre(inputs)

        assert torch.equal(outputs, expected_outputs)

    def test_post(self):
        post = nn.Linear(2, 3)
        block = Block(post=post)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = module_utils.module_test(block, inputs)
        expected_outputs = post(inputs)

        assert torch.equal(outputs, expected_outputs)

    def test_pre_post(self):
        pre = nn.Linear(2, 3)
        post = nn.Linear(3, 4)
        block = Block(pre=pre, post=post)
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = module_utils.module_test(block, inputs)
        expected_outputs = pre(inputs)
        expected_outputs = post(expected_outputs)

        assert torch.equal(outputs, expected_outputs)


import inspect
from typing import Final


def is_tabular_module(module: torch.nn.Module) -> bool:
    # Get the forward method of the input module
    forward_method = module.forward

    # Get the signature of the forward method
    forward_signature = inspect.signature(forward_method)

    # Get the first argument of the forward method
    first_arg = list(forward_signature.parameters.values())[0]

    # Check if the annotation exists for the first argument
    if first_arg.annotation != inspect.Parameter.empty:
        # Check if the annotation is a dict of tensors
        if first_arg.annotation == Dict[str, torch.Tensor]:
            return True

    return False


class Sequential(nn.Sequential):
    accepts_dict: Final[bool]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepts_dict = is_tabular_module(self[0])

    def forward(self, input: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        if not self.accepts_dict:
            if not torch.jit.isinstance(input, torch.Tensor):
                raise RuntimeError("Expected a tensor, but got a dictionary instead.")
            x: torch.Tensor = input
            for module in self:
                x = module(x)

            return x

        if not torch.jit.isinstance(input, Dict[str, torch.Tensor]):
            raise RuntimeError(f"Expected a dictionary of tensors, but got {type(input)} instead.")

        x: Dict[str, torch.Tensor] = input
        for module in self:
            x = module(x)

        return x


class TestTabularBlock:
    def test_no_pre_post_aggregation(self):
        block = TabularBlock()
        inputs = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

        outputs = module_utils.module_test(block, inputs)

        assert torch.equal(inputs["a"], outputs["a"])

    def test_aggregation(self):
        aggregation = ConcatDict()
        block = TabularBlock(agg=aggregation)
        inputs = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

        outputs = module_utils.module_test(block, inputs)
        expected_outputs = aggregation(inputs)

        assert torch.equal(outputs, expected_outputs)

    def test_sequential(self):
        inputs = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

        custom = Sequential(TabularBlock(), TabularBlock())

        seq = nn.Sequential(TabularBlock(), TabularBlock())
        to_call = TabularBlock(seq, agg=ConcatDict())

        outputs = module_utils.module_test(to_call, inputs)

        torch.equal(outputs, inputs["a"])
