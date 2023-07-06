import pytest
import torch

from merlin.models.torch.transforms.bias import LogitsTemperatureScaler
from merlin.models.torch.utils import module_utils


class TestLogitsTemperatureScaler:
    def test_init(self):
        """Test correct temperature initialization."""
        scaler = LogitsTemperatureScaler(0.5)
        assert scaler.temperature == 0.5

    def test_invalid_temperature_type(self):
        """Test exception is raised for incorrect temperature type."""
        with pytest.raises(ValueError, match=r"Invalid temperature type"):
            LogitsTemperatureScaler("invalid")

    def test_invalid_temperature_value(self):
        """Test exception is raised for out-of-range temperature value."""
        with pytest.raises(ValueError, match=r"Invalid temperature value"):
            LogitsTemperatureScaler(1.5)

    def test_temperature_scaling(self):
        """Test temperature scaling of logits."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        expected_scaled_logits = torch.tensor([2.0, 4.0, 6.0])

        scaler = LogitsTemperatureScaler(0.5)
        outputs = module_utils.module_test(scaler, logits)
        assert torch.allclose(outputs, expected_scaled_logits)

    def test_zero_temperature_value(self):
        """Test exception is raised for zero temperature value."""
        with pytest.raises(ValueError, match=r"Invalid temperature value"):
            LogitsTemperatureScaler(0.0)
