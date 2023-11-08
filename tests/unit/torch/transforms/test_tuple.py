import pytest
import torch

from merlin.models.torch.transforms import tuple
from merlin.models.torch.utils import module_utils
from merlin.schema import Schema


class TestToTuple:
    @pytest.mark.parametrize("length", [i + 1 for i in range(10)])
    def test_with_length(self, length):
        schema = Schema([str(i) for i in range(length)])
        to_tuple = tuple.ToTuple(schema)
        assert isinstance(to_tuple, getattr(tuple, f"ToTuple{length}"))

        inputs = {str(i): torch.randn(2, 3) for i in range(length)}
        outputs = module_utils.module_test(to_tuple, inputs)

        assert len(outputs) == length

    def test_exception(self):
        with pytest.raises(ValueError):
            tuple.ToTuple(Schema([str(i) for i in range(11)]))

        to_tuple = tuple.ToTuple2()
        inputs = {"0": torch.randn(2, 3), "1": torch.randn(2, 3)}
        with pytest.raises(RuntimeError):
            module_utils.module_test(to_tuple, inputs)
