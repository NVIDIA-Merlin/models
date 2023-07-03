import torch

import merlin.models.torch as mm
from merlin.models.torch.blocks.experts import (  # CGCBlock,; PLEBlock,; PLEExpertGateBlock,
    ExpertGateBlock,
    MMOEBlock,
)
from merlin.models.torch.utils import module_utils


class TestMMOEBlock:
    def test_init(self):
        mmoe = MMOEBlock(mm.MLPBlock([10, 10]), 2)

        assert isinstance(mmoe, MMOEBlock)
        assert isinstance(mmoe[0], mm.ShortcutBlock)
        assert len(mmoe[0][0].branches) == 2
        for i in range(2):
            assert mmoe[0][0][str(i)][1].out_features == 10
            assert mmoe[0][0][str(i)][3].out_features == 10
        assert isinstance(mmoe[0][0].post[0], mm.Stack)
        assert isinstance(mmoe[1], ExpertGateBlock)
        assert mmoe[1][0][0].out_features == 1

    def test_init_with_outputs(self):
        outputs = mm.ParallelBlock({"a": mm.BinaryOutput(), "b": mm.BinaryOutput()})
        outputs.prepend_for_each(mm.MLPBlock([10, 10]))
        outputs.prepend(MMOEBlock(mm.MLPBlock([10, 10]), 2, outputs))

        assert isinstance(outputs.pre[0], MMOEBlock)
        assert list(outputs.pre[0][1].keys()) == ["a", "b"]

    def test_forward(self):
        mmoe = MMOEBlock(mm.MLPBlock([2, 2]), 2)

        outputs = module_utils.module_test(mmoe, torch.rand(5, 5))
        assert outputs.shape == (5, 10)

    def test_forward_with_outputs(self):
        outputs = mm.ParallelBlock({"a": mm.BinaryOutput(), "b": mm.BinaryOutput()})
        outputs.prepend_for_each(mm.MLPBlock([2, 2]))
        outputs.prepend(MMOEBlock(mm.MLPBlock([2, 2]), 2, outputs))

        outputs = module_utils.module_test(outputs, torch.rand(5, 5))
        assert outputs["a"].shape == (5, 1)
        assert outputs["b"].shape == (5, 1)
