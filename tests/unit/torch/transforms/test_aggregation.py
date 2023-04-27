# import pytest
# import torch

# from merlin.models.torch.base import TabularBlock
# from merlin.models.torch.transforms.aggregation import ConcatFeatures, StackFeatures, SumResidual


# class TestConcatFeatures:
#     def test_valid_input(self):
#         concat = ConcatFeatures(dim=1)
#         input_tensors = {
#             "a": torch.randn(2, 3),
#             "b": torch.randn(2, 4),
#         }
#         output = concat(input_tensors)
#         assert output.shape == (2, 7)

#     def test_same_order(self):
#         concat = ConcatFeatures(dim=2)
#         a = torch.randn(2, 3, 4)
#         b = torch.randn(2, 3, 5)
#         output_a = concat({"a": a, "b": b})
#         output_b = concat(
#             {
#                 "b": b,
#                 "a": a,
#             }
#         )

#         assert torch.all(torch.eq(output_a, output_b))

#     def test_invalid_input(self):
#         concat = ConcatFeatures(dim=1)
#         input_tensors = {
#             "a": torch.randn(2, 3),
#             "b": torch.randn(3, 3),
#         }
#         with pytest.raises(RuntimeError, match="Input tensor shapes don't match"):
#             concat(input_tensors)

#     def test_as_aggregation_string(self):
#         block = TabularBlock(aggregation="concat")

#         input_tensors = {
#             "a": torch.randn(2, 3),
#             "b": torch.randn(2, 4),
#         }
#         output = block(input_tensors)
#         assert output.shape == (2, 7)


# class TestStackFeatures:
#     def test_valid_input(self):
#         stack = StackFeatures(dim=0)
#         input_tensors = {
#             "a": torch.randn(2, 3),
#             "b": torch.randn(2, 3),
#         }
#         output = stack(input_tensors)
#         assert output.shape == (2, 2, 3)

#     def test_same_order(self):
#         stack = StackFeatures(dim=0)
#         a = torch.randn(2, 3)
#         b = torch.randn(2, 3)
#         output_a = stack({"a": a, "b": b})
#         output_b = stack(
#             {
#                 "b": b,
#                 "a": a,
#             }
#         )

#         assert torch.all(torch.eq(output_a, output_b))

#     def test_invalid_input(self):
#         stack = StackFeatures(dim=0)
#         input_tensors = {
#             "a": torch.randn(2, 3),
#             "b": torch.randn(3, 3),
#         }
#         with pytest.raises(RuntimeError, match="Input tensor shapes don't match"):
#             stack(input_tensors)

#     def test_as_aggregation_string(self):
#         block = TabularBlock(aggregation="stack")

#         input_tensors = {
#             "a": torch.randn(2, 3),
#             "b": torch.randn(2, 3),
#         }
#         output = block(input_tensors)
#         assert output.shape == (2, 2, 3)


# class TestSumResidual:
#     def test_single_output(self):
#         sum_residual = SumResidual()

#         inputs = {
#             "shortcut": torch.tensor([1.0, 2.0, 3.0]),
#             "input_1": torch.tensor([4.0, 5.0, 6.0]),
#         }

#         output = sum_residual(inputs)
#         expected_output = torch.tensor([5.0, 7.0, 9.0])

#         assert torch.allclose(output, expected_output)

#     def test_multiple_outputs(self):
#         sum_residual = SumResidual(activation="sigmoid")

#         inputs = {
#             "shortcut": torch.tensor([1.0, 2.0, 3.0]),
#             "input_1": torch.tensor([4.0, 5.0, 6.0]),
#             "input_2": torch.tensor([7.0, 8.0, 9.0]),
#         }

#         outputs = sum_residual(inputs)

#         expected_output_1 = torch.sigmoid(torch.tensor([5.0, 7.0, 9.0]))
#         expected_output_2 = torch.sigmoid(torch.tensor([8.0, 10.0, 12.0]))

#         assert torch.allclose(outputs["input_1"], expected_output_1)
#         assert torch.allclose(outputs["input_2"], expected_output_2)

#     def test_no_shortcut(self):
#         sum_residual = SumResidual()

#         inputs = {
#             "input_1": torch.tensor([4.0, 5.0, 6.0]),
#         }

#         with pytest.raises(
#             RuntimeError, match="Shortcut 'shortcut' not found in the inputs dictionary"
#         ):
#             sum_residual(inputs)

#     def test_invalid_activation(self):
#         with pytest.raises(
#             ValueError,
#             match="torch does not have the specified activation function: invalid_activation",
#         ):
#             SumResidual(activation="invalid_activation")

#     def test_as_aggregation_string(self):
#         sum_residual = TabularBlock(aggregation="sum-residual")

#         inputs = {
#             "shortcut": torch.tensor([1.0, 2.0, 3.0]),
#             "input_1": torch.tensor([4.0, 5.0, 6.0]),
#         }

#         output = sum_residual(inputs)
#         expected_output = torch.tensor([5.0, 7.0, 9.0])

#         assert torch.allclose(output, expected_output)
