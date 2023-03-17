import pytest
import torch

from merlin.models.torch.core.aggregation import SumResidual


class TestSumResidual:
    def test_single_output(self):
        sum_residual = SumResidual()

        inputs = {
            "shortcut": torch.tensor([1.0, 2.0, 3.0]),
            "input_1": torch.tensor([4.0, 5.0, 6.0]),
        }

        output = sum_residual(inputs)
        expected_output = torch.tensor([5.0, 7.0, 9.0])

        assert torch.allclose(output, expected_output)

    def test_multiple_outputs(self):
        sum_residual = SumResidual(activation="sigmoid")

        inputs = {
            "shortcut": torch.tensor([1.0, 2.0, 3.0]),
            "input_1": torch.tensor([4.0, 5.0, 6.0]),
            "input_2": torch.tensor([7.0, 8.0, 9.0]),
        }

        outputs = sum_residual(inputs)

        expected_output_1 = torch.sigmoid(torch.tensor([5.0, 7.0, 9.0]))
        expected_output_2 = torch.sigmoid(torch.tensor([8.0, 10.0, 12.0]))

        assert torch.allclose(outputs["input_1"], expected_output_1)
        assert torch.allclose(outputs["input_2"], expected_output_2)

    def test_no_shortcut(self):
        sum_residual = SumResidual()

        inputs = {
            "input_1": torch.tensor([4.0, 5.0, 6.0]),
        }

        with pytest.raises(
            ValueError, match="Shortcut 'shortcut' not found in the inputs dictionary"
        ):
            sum_residual(inputs)

    def test_invalid_activation(self):
        with pytest.raises(
            ValueError,
            match="torch does not have the specified activation function: invalid_activation",
        ):
            SumResidual(activation="invalid_activation")
