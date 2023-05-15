import pytest
import torch

from merlin.models.torch.block import Block
from merlin.models.torch.transforms.agg import Concat, MaybeAgg, Stack
from merlin.models.torch.utils import module_utils


class TestConcat:
    def test_valid_input(self):
        concat = Concat(dim=1)
        input_tensors = {
            "a": torch.randn(2, 3),
            "b": torch.randn(2, 4),
        }
        output = module_utils.module_test(concat, input_tensors)
        assert output.shape == (2, 7)

    @pytest.mark.parametrize("dim", [2, -1])
    def test_same_order(self, dim):
        concat = Concat(dim=dim)
        a = torch.randn(2, 3, 4)
        b = torch.randn(2, 3, 5)
        output_a = module_utils.module_test(concat, {"a": a, "b": b})
        output_b = module_utils.module_test(concat, {"b": b, "a": a})

        assert torch.all(torch.eq(output_a, output_b))
        assert output_a.shape == (2, 3, 9)

    def test_invalid_input(self):
        concat = Concat(dim=1)
        input_tensors = {
            "a": torch.randn(2, 3),
            "b": torch.randn(3, 3),
        }
        with pytest.raises(RuntimeError, match="Input tensor shapes don't match"):
            concat(input_tensors)

    def test_from_registry(self):
        block = Block.parse("concat")

        input_tensors = {
            "a": torch.randn(2, 3),
            "b": torch.randn(2, 4),
        }
        output = module_utils.module_test(block, input_tensors)
        assert output.shape == (2, 7)


class TestStack:
    def test_2d_input(self):
        stack = Stack(dim=0)
        input_tensors = {
            "a": torch.randn(2, 3),
            "b": torch.randn(2, 3),
        }
        output = module_utils.module_test(stack, input_tensors)
        assert output.shape == (2, 2, 3)

    def test_same_order(self):
        stack = Stack(dim=0)
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        output_a = module_utils.module_test(stack, {"a": a, "b": b})
        output_b = module_utils.module_test(stack, {"b": b, "a": a})

        assert torch.all(torch.eq(output_a, output_b))

    def test_invalid_input(self):
        stack = Stack(dim=0)
        input_tensors = {
            "a": torch.randn(2, 3),
            "b": torch.randn(3, 3),
        }
        with pytest.raises(RuntimeError, match="Input tensor shapes don't match"):
            stack(input_tensors)

    def test_from_registry(self):
        block = Block.parse("stack")

        input_tensors = {
            "a": torch.randn(2, 3),
            "b": torch.randn(2, 3),
        }
        output = block(input_tensors)
        assert output.shape == (2, 2, 3)


class TestMaybeAgg:
    def test_with_single_tensor(self):
        tensor = torch.tensor([1, 2, 3])
        stack = Stack(dim=0)
        maybe_agg = MaybeAgg(agg=stack)

        output = module_utils.module_test(maybe_agg, tensor)
        assert torch.equal(output, tensor)

    def test_with_dict(self):
        stack = Stack(dim=0)
        maybe_agg = MaybeAgg(agg=stack)

        tensor1 = torch.tensor([[1, 2], [3, 4]])
        tensor2 = torch.tensor([[5, 6], [7, 8]])
        input_dict = {"tensor1": tensor1, "tensor2": tensor2}
        expected_output = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        output = module_utils.module_test(maybe_agg, input_dict)

        assert torch.equal(output, expected_output)

    def test_with_incompatible_dict(self):
        concat = Concat(dim=0)
        maybe_agg = MaybeAgg(agg=concat)

        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([4, 5])
        input_dict = {"tensor1": (tensor1, tensor2)}

        with pytest.raises(
            RuntimeError, match="Inputs must be either a dictionary of tensors or a single tensor"
        ):
            maybe_agg(input_dict)
