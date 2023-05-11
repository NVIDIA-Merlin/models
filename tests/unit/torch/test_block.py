import torch
from torch import nn

from merlin.models.torch.batch import Batch
from merlin.models.torch.block import Block
from merlin.models.torch.utils import module_utils


class PlusOne(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + 1


class TestBlock:
    def test_identity(self):
        block = Block()

        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        outputs = module_utils.module_test(block, inputs, batch=Batch(inputs))

        assert torch.equal(inputs, outputs)

    def test_insertion(self):
        block = Block()
        block.prepend(PlusOne())
        block.append(PlusOne())

        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        outputs = module_utils.module_test(block, inputs, batch=Batch(inputs))

        assert torch.equal(outputs, inputs + 2)

    def test_copy(self):
        block = Block(PlusOne())

        assert isinstance(block.copy(), Block)
        assert isinstance(block.copy()[0], PlusOne)
        assert block.copy() != block

    def test_repeat(self):
        block = Block(PlusOne())

        assert isinstance(block.repeat(2), Block)
        assert len(block.repeat(2)) == 2
