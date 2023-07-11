import pytest
import torch

import merlin.models.torch as mm
from merlin.models.torch.blocks.experts import (
    CGCBlock,
    ExpertGateBlock,
    MMOEBlock,
    PLEBlock,
    PLEExpertGateBlock,
)
from merlin.models.torch.utils import module_utils

dict_inputs = {"experts": torch.rand((10, 4, 5)), "shortcut": torch.rand((10, 8))}


class TestExpertGateBlock:
    @pytest.fixture
    def expert_gate(self):
        return ExpertGateBlock(num_experts=4)

    def test_requires_dict_input(self, expert_gate):
        with pytest.raises(RuntimeError, match="ExpertGateBlock requires a dictionary input"):
            expert_gate(torch.rand((10, 5)))

    def test_forward_pass(self, expert_gate):
        result = module_utils.module_test(expert_gate, dict_inputs)
        assert result.shape == (10, 5)


class TestMMOEBlock:
    def test_init(self):
        mmoe = MMOEBlock(mm.MLPBlock([2, 2]), 2)

        assert isinstance(mmoe, MMOEBlock)
        assert isinstance(mmoe[0], mm.ShortcutBlock)
        assert len(mmoe[0][0].branches) == 2
        for i in range(2):
            assert mmoe[0][0][str(i)][1].out_features == 2
            assert mmoe[0][0][str(i)][3].out_features == 2
        assert isinstance(mmoe[0][0].post[0], mm.Stack)
        assert isinstance(mmoe[1], ExpertGateBlock)
        assert mmoe[1][0][0].out_features == 2

    def test_init_with_outputs(self):
        outputs = mm.ParallelBlock({"a": mm.BinaryOutput(), "b": mm.BinaryOutput()})
        outputs.prepend_for_each(mm.MLPBlock([2]))
        outputs.prepend(MMOEBlock(mm.MLPBlock([2, 2]), 2, outputs))

        assert isinstance(outputs.pre[0], MMOEBlock)
        assert list(outputs.pre[0][1].keys()) == ["a", "b"]

    def test_forward(self):
        mmoe = MMOEBlock(mm.MLPBlock([2, 2]), 2)

        outputs = module_utils.module_test(mmoe, torch.rand(5, 5))
        assert outputs.shape == (5, 2)

    def test_forward_with_outputs(self):
        outputs = mm.ParallelBlock({"a": mm.BinaryOutput(), "b": mm.BinaryOutput()})
        outputs.prepend_for_each(mm.MLPBlock([2, 2]))
        outputs.prepend(MMOEBlock(mm.MLPBlock([2, 2]), 2, outputs))

        outputs = module_utils.module_test(outputs, torch.rand(5, 5))
        assert outputs["a"].shape == (5, 1)
        assert outputs["b"].shape == (5, 1)


class TestPLEExpertGateBlock:
    @pytest.fixture
    def ple_expert_gate(self):
        return PLEExpertGateBlock(
            num_experts=6, task_experts=mm.repeat_parallel(mm.MLPBlock([5, 5]), 2), name="a"
        )

    def test_repr(self, ple_expert_gate):
        assert "(task_experts)" in str(ple_expert_gate)
        assert "(gate)" in str(ple_expert_gate)

    def test_requires_dict_input(self, ple_expert_gate):
        with pytest.raises(RuntimeError, match="ExpertGateBlock requires a dictionary input"):
            ple_expert_gate(torch.rand((10, 5)))

    def test_ple_forward(self, ple_expert_gate):
        result = module_utils.module_test(ple_expert_gate, dict_inputs)
        assert result.shape == (10, 5)


class TestCGCBlock:
    @pytest.mark.parametrize("shared_gate", [True, False])
    def test_forward(self, music_streaming_data, shared_gate):
        output_block = mm.TabularOutputBlock(music_streaming_data.schema, init="defaults")
        cgc = CGCBlock(
            mm.MLPBlock([5]),
            num_shared_experts=2,
            num_task_experts=2,
            outputs=output_block,
            shared_gate=shared_gate,
        )

        outputs = module_utils.module_test(cgc, torch.rand(5, 5))
        assert len(outputs) == len(output_block) + (2 if shared_gate else 0)


class TestPLEBlock:
    def test_forward(self, music_streaming_data):
        output_block = mm.TabularOutputBlock(music_streaming_data.schema, init="defaults")
        ple = PLEBlock(
            mm.MLPBlock([5]),
            num_shared_experts=2,
            num_task_experts=2,
            depth=2,
            outputs=output_block,
        )

        assert isinstance(ple[0], CGCBlock)
        assert len(ple[0][1]) == len(output_block) + 1
        assert isinstance(ple[0][1]["experts"][0], ExpertGateBlock)
        assert isinstance(ple[1], CGCBlock)
        assert list(ple[1][1].branches.keys()) == list(ple[1][1].branches.keys())

        outputs = module_utils.module_test(ple, torch.rand(5, 5))
        assert len(outputs) == len(output_block)
