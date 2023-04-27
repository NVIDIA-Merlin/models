# import pytest
# import torch

# from merlin.models.torch.utils import padding_utils


# class Test_get_last:
#     # Test cases for get_last_elements_from_padded_tensor
#     @pytest.mark.parametrize(
#         "padded_tensor, expected_last_elements",
#         [
#             (
#                 torch.tensor(
#                     [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [6, 7, 8, 9, 0]], dtype=torch.float32
#                 ),
#                 torch.tensor([3.0, 5.0, 9.0]),
#             ),
#             (
#                 torch.tensor(
#                     [[1, 2, 3, 4, 5], [6, 7, 8, 0, 0], [9, 10, 0, 0, 0]], dtype=torch.float32
#                 ),
#                 torch.tensor([5.0, 8.0, 10.0]),
#             ),
#             (
#                 torch.tensor(
#                     [[1, 0, 0, 0, 0], [2, 3, 0, 0, 0], [4, 5, 6, 0, 0]], dtype=torch.float32
#                 ),
#                 torch.tensor([1.0, 3.0, 6.0]),
#             ),
#         ],
#     )
#     def test_get_last_elements_from_padded_tensor(self, padded_tensor, expected_last_elements):
#         last_elements = padding_utils.get_last(padded_tensor)
#         assert torch.all(last_elements == expected_last_elements)


# class Test_dict_remove_last:
#     def test_single_tensor(self):
#         tensor_a = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]])
#         tensor_dict = {"a": tensor_a}

#         result = padding_utils.remove_last_non_padded(tensor_dict)

#         expected_transformed_a = torch.tensor([[1, 2, 0], [4, 0, 0], [6, 7, 8]])

#         assert torch.equal(result["a"], expected_transformed_a)

#     def test_multiple_tensors(self):
#         tensor_a = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]])
#         tensor_b = torch.tensor([[11, 12, 13, 0], [14, 15, 0, 0], [16, 17, 18, 19]])
#         tensor_dict = {"a": tensor_a, "b": tensor_b}

#         result = padding_utils.remove_last_non_padded(tensor_dict)

#         expected_transformed_a = torch.tensor([[1, 2, 0], [4, 0, 0], [6, 7, 8]])
#         expected_transformed_b = torch.tensor([[11, 12, 0], [14, 0, 0], [16, 17, 18]])

#         assert torch.equal(result["a"], expected_transformed_a)
#         assert torch.equal(result["b"], expected_transformed_b)
