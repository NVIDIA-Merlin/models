from typing import Dict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional

# from merlin.models.torch.base import Block, TabularBlock
from merlin.models.torch.base import Block, ParallelBlock, TabularBlock, TabularIdentity
from merlin.models.torch.utils import module_utils
from merlin.schema import Schema


class ConcatDict(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(x.values()), dim=-1)


class TabularMultiply(nn.Module):
    def __init__(self, num: int, suffix: str = ""):
        super().__init__()
        self.num = num
        self.suffix = suffix

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key + self.suffix: val * self.num for key, val in inputs.items()}


class TestBlock:
    def test_no_pre_post(self):
        block = Block()
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        outputs = module_utils.module_test(block, inputs)

        assert torch.equal(inputs, outputs)

    def test_no_pre_post_tabular(self):
        block = Block(TabularIdentity())
        inputs = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

        outputs = module_utils.module_test(block, inputs)

        assert torch.equal(inputs["a"], outputs["a"])

    def test_no_pre_tabular(self):
        block = Block(TabularIdentity(), post=ConcatDict())
        inputs = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

        outputs = module_utils.module_test(block, inputs)

        assert torch.equal(inputs["a"], outputs)

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


class TestParallelBlock:
    def test_single_branch(self):
        linear = nn.Linear(2, 3)
        parallel_block = ParallelBlock(linear)
        x = torch.randn(4, 2)
        out = module_utils.module_test(parallel_block, x)

        assert isinstance(out, dict)
        assert len(out) == 1
        assert torch.allclose(out["0"], linear(x))

    def test_single_branch_dict(self):
        linear = TabularIdentity()
        parallel_block = ParallelBlock(linear)
        x = {"a": torch.randn(4, 2)}
        out = module_utils.module_test(parallel_block, x)

        assert isinstance(out, dict)
        assert len(out) == 1
        assert torch.allclose(out["a"], x["a"])

    def test_branch_list(self):
        layers = [nn.Linear(2, 3), nn.ReLU(), nn.Linear(2, 1)]
        parallel_block = ParallelBlock(*layers)
        x = torch.randn(4, 2)
        out = module_utils.module_test(parallel_block, x)

        assert isinstance(out, dict)
        assert len(out) == len(layers)
        assert set(out.keys()) == {"0", "1", "2"}

        for i, layer in enumerate(layers):
            assert torch.allclose(out[str(i)], layer(x))

    def test_branch_list_dict(self):
        layers = [TabularMultiply(1, "1"), TabularMultiply(2, "2"), TabularMultiply(3, "3")]
        parallel_block = ParallelBlock(*layers)
        x = {"a": torch.randn(4, 2)}
        out = module_utils.module_test(parallel_block, x)

        assert isinstance(out, dict)
        assert len(out) == len(layers)
        assert set(out.keys()) == {"a1", "a2", "a3"}

        for i, layer in enumerate(layers):
            assert torch.allclose(out["a" + str(i + 1)], layer(x)["a" + str(i + 1)])

    def test_branch_list_dict_same_keys(self):
        layers = [TabularMultiply(1), TabularMultiply(2)]
        parallel_block = ParallelBlock(*layers)
        x = {"a": torch.randn(4, 2)}

        with pytest.raises(RuntimeError):
            parallel_block(x)

    def test_branch_dict(self):
        layers_dict = {"linear": nn.Linear(2, 3), "relu": nn.ReLU(), "linear2": nn.Linear(2, 1)}
        parallel_block = ParallelBlock(layers_dict)
        x = torch.randn(4, 2)
        out = module_utils.module_test(parallel_block, x)

        assert isinstance(out, dict)
        assert len(out) == len(layers_dict)
        assert list(out.keys()) == list(layers_dict.keys())

        for name, layer in layers_dict.items():
            assert torch.allclose(out[name], layer(x))


# class TestTabularBlock:
#     def test_no_pre_post_aggregation(self):
#         block = TabularBlock()
#         inputs = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

#         outputs = module_utils.module_test(block, inputs)

#         assert torch.equal(inputs["a"], outputs["a"])

#     def test_aggregation(self):
#         aggregation = ConcatDict()
#         block = TabularBlock(agg=aggregation)
#         inputs = {"a": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

#         outputs = module_utils.module_test(block, inputs)
#         expected_outputs = aggregation(inputs)

#         assert torch.equal(outputs, expected_outputs)
