import pytest
import torch

from merlin.models.torch.outputs.base import DotProduct


class TestDotProduct:
    def test_output_shape(self):
        dp = DotProduct()

        query = torch.randn(4, 10)
        candidate = torch.randn(4, 10)
        inputs = {"query": query, "candidate": candidate}

        output = dp(inputs)
        assert output.shape == (4, 1), "Output shape mismatch"

    def test_output_values(self):
        dp = DotProduct()

        query = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        candidate = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        inputs = {"query": query, "candidate": candidate}

        output = dp(inputs)
        expected_output = torch.tensor([[1], [4]], dtype=torch.float32)
        assert torch.allclose(output, expected_output), "Output values mismatch"

    def test_custom_names(self):
        dp = DotProduct(query_name="user", item_name="product")

        query = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        candidate = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        inputs = {"user": query, "product": candidate}

        output = dp(inputs)
        expected_output = torch.tensor([[1], [4]], dtype=torch.float32)
        assert torch.allclose(output, expected_output), "Output values mismatch with custom names"

    @pytest.mark.parametrize("query_shape, candidate_shape", [((4, 5), (4, 6)), ((3, 5), (4, 5))])
    def test_dotproduct_input_shape_mismatch(self, query_shape, candidate_shape):
        dp = DotProduct()

        query = torch.randn(*query_shape)
        candidate = torch.randn(*candidate_shape)
        inputs = {"query": query, "candidate": candidate}

        with pytest.raises(RuntimeError):
            dp(inputs)

    def test_missing_key(self):
        dp = DotProduct()

        query = torch.randn(4, 10)
        inputs = {"query": query}

        with pytest.raises(RuntimeError) as exc_info:
            dp(inputs)

        expected_msg = (
            "Key 'candidate' not found in input dictionary. "
            "Please provide a dictionary with keys 'query' and 'candidate'."
        )
        assert str(exc_info.value) == expected_msg

    def test_invalid_input_type(self):
        dp = DotProduct()

        inputs = "invalid"

        with pytest.raises(RuntimeError) as exc_info:
            dp(inputs)

        expected_msg = (
            "Invalid input type. Expected inputs to be either a dictionary with keys "
            "'query' and 'candidate' or a tuple/list of size 2."
        )
        assert str(exc_info.value) == expected_msg

    def test_invalid_tuple_size(self):
        dp = DotProduct()

        query = torch.randn(4, 10)
        candidate = torch.randn(4, 10)
        inputs = (query, candidate, "extra")

        with pytest.raises(RuntimeError) as exc_info:
            dp(inputs)

        expected_msg = (
            "Invalid input type. Expected inputs to be either a dictionary with keys "
            "'query' and 'candidate' or a tuple/list of size 2."
        )
        assert str(exc_info.value) == expected_msg
