import pytest
import torch

from merlin.models.torch.transforms import tuple
from merlin.models.torch.utils import module_utils


class TestToTuple:
    @pytest.mark.parametrize("length", [i + 1 for i in range(10)])
    def test_with_length(self, length):
        to_tuple = getattr(tuple, f"ToTuple{length}")()

        inputs = {str(i): torch.randn(2, 3) for i in range(length)}
        outputs = module_utils.module_test(to_tuple, inputs)

        assert len(outputs) == length
