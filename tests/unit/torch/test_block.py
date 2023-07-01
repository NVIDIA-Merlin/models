#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Dict, Tuple

import pytest
import torch
from torch import nn

import merlin.models.torch as mm
from merlin.models.torch.batch import Batch
from merlin.models.torch.block import (
    Block,
    ParallelBlock,
    ResidualBlock,
    ShortcutBlock,
    get_pre,
    set_pre,
)
from merlin.models.torch.container import BlockContainer, BlockContainerDict
from merlin.models.torch.utils import module_utils
from merlin.schema import Tags


class PlusOne(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + 1


class PlusOneDict(nn.Module):
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v + 1 for k, v in inputs.items()}


class PlusOneTuple(nn.Module):
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs + 1, inputs + 1


class TestBlock:
    def test_identity(self):
        block = Block()

        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        outputs = module_utils.module_test(block, inputs, batch=Batch(inputs))

        assert torch.equal(inputs, outputs)
        assert mm.schema.output(block) == mm.schema.output.tensors(inputs)

    def test_insertion(self):
        block = Block()
        block.prepend(PlusOne())
        block.append(PlusOne())

        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        outputs = module_utils.module_test(block, inputs, batch=Batch(inputs))

        assert torch.equal(outputs, inputs + 2)

        # block.append(PlusOne(), link="residual")
        # assert isinstance(block[-1], link.Residual)

    def test_copy(self):
        block = Block(PlusOne())

        copied = block.copy()
        assert isinstance(copied, Block)
        assert isinstance(copied[0], PlusOne)
        assert copied != block

        copied.some_attribute = "new value"
        assert not hasattr(block, "some_attribute")

    def test_repeat(self):
        block = Block(PlusOne())

        assert isinstance(block.repeat(2), Block)
        assert len(block.repeat(2)) == 2

        with pytest.raises(TypeError, match="n must be an integer"):
            block.repeat("invalid_input")

        with pytest.raises(ValueError, match="n must be greater than 0"):
            block.repeat(0)

    # def test_repeat_with_link(self):
    #     block = Block(PlusOne())

    #     repeated = block.repeat(2, link="residual")
    #     assert isinstance(repeated, Block)
    #     assert len(repeated) == 2
    #     assert isinstance(repeated[-1], link.Residual)

    #     inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    #     outputs = module_utils.module_test(repeated, inputs)

    #     assert torch.equal(outputs, (inputs + 1) + (inputs + 1) + 1)

    def test_from_registry(self):
        @Block.registry.register("my_block")
        class MyBlock(Block):
            def forward(self, inputs):
                _inputs = inputs + 1

                return super().forward(_inputs)

        block = Block.parse("my_block")
        assert block.__class__ == MyBlock

        inputs = torch.randn(1, 3)
        assert torch.equal(block(inputs), inputs + 1)


class TestParallelBlock:
    def test_init(self):
        pb = ParallelBlock({"test": PlusOne()})
        assert isinstance(pb, ParallelBlock)
        assert isinstance(pb.pre, BlockContainer)
        assert isinstance(pb.branches, BlockContainerDict)
        assert isinstance(pb.post, BlockContainer)
        assert pb.__repr__().startswith("ParallelBlock")

    def test_init_list_of_dict(self):
        pb = ParallelBlock(({"test": PlusOne()}))
        assert len(pb) == 1
        assert "test" in pb

    def test_forward(self):
        inputs = torch.randn(1, 3)
        pb = ParallelBlock({"test": PlusOne()})
        outputs = module_utils.module_test(pb, inputs)
        assert isinstance(outputs, dict)
        assert "test" in outputs

    def test_forward_dict(self):
        inputs = {"a": torch.randn(1, 3)}
        pb = ParallelBlock({"test": PlusOneDict()})
        outputs = module_utils.module_test(pb, inputs)
        assert isinstance(outputs, dict)
        assert "a" in outputs

    def test_forward_dict_duplicate(self):
        inputs = {"a": torch.randn(1, 3)}
        pb = ParallelBlock({"1": PlusOneDict(), "2": PlusOneDict()})

        with pytest.raises(RuntimeError):
            pb(inputs)

    def test_forward_tensor_duplicate(self):
        class PlusOneKey(nn.Module):
            def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
                return inputs["2"] + 1

        pb = ParallelBlock({"1": PlusOneDict(), "2": PlusOneKey()})
        inputs = {"2": torch.randn(1, 3)}

        with pytest.raises(RuntimeError):
            pb(inputs)

    def test_schema_tracking(self):
        pb = ParallelBlock({"a": PlusOne(), "b": PlusOne()})

        inputs = torch.randn(1, 3)
        outputs = mm.schema.trace(pb, inputs)
        schema = mm.schema.output(pb)

        for name in outputs:
            assert name in schema.column_names
            assert schema[name].dtype.name == str(outputs[name].dtype).split(".")[-1]

        assert len(schema.select_by_tag(Tags.EMBEDDING)) == 2

    def test_forward_tuple(self):
        inputs = torch.randn(1, 3)
        pb = ParallelBlock({"test": PlusOneTuple()})
        with pytest.raises(RuntimeError):
            module_utils.module_test(pb, inputs)

    def test_append(self):
        module = PlusOneDict()
        pb = ParallelBlock({"test": PlusOne()})
        pb.append(module)
        assert len(pb.post._modules) == 1

        assert pb[-1][0] == module
        assert pb[2][0] == module

        repr = pb.__repr__()
        assert "(post):" in repr

        module_utils.module_test(pb, torch.randn(1, 3))

    def test_prepend(self):
        module = PlusOne()
        pb = ParallelBlock({"test": module})
        pb.prepend(module)
        assert len(pb.pre._modules) == 1

        assert pb[0][0] == module

        repr = pb.__repr__()
        assert "(pre):" in repr

        module_utils.module_test(pb, torch.randn(1, 3))

    def test_append_to(self):
        module = nn.Module()
        pb = ParallelBlock({"test": module})
        pb.append_to("test", module)
        assert len(pb["test"]) == 2

    def test_prepend_to(self):
        module = nn.Module()
        pb = ParallelBlock({"test": module})
        pb.prepend_to("test", module)
        assert len(pb["test"]) == 2

    def test_append_for_each(self):
        module = nn.Module()
        pb = ParallelBlock({"a": module, "b": module})
        pb.append_for_each(module)
        assert len(pb["a"]) == 2
        assert len(pb["b"]) == 2
        assert pb["a"][-1] != pb["b"][-1]

        pb.append_for_each(module, shared=True)
        assert len(pb["a"]) == 3
        assert len(pb["b"]) == 3
        assert pb["a"][-1] == pb["b"][-1]

    def test_prepend_for_each(self):
        module = nn.Module()
        pb = ParallelBlock({"a": module, "b": module})
        pb.prepend_for_each(module)
        assert len(pb["a"]) == 2
        assert len(pb["b"]) == 2
        assert pb["a"][0] != pb["b"][0]

        pb.prepend_for_each(module, shared=True)
        assert len(pb["a"]) == 3
        assert len(pb["b"]) == 3
        assert pb["a"][0] == pb["b"][0]

    def test_getitem(self):
        module = nn.Module()
        pb = ParallelBlock({"test": module})
        assert isinstance(pb["test"], BlockContainer)

        with pytest.raises(IndexError):
            pb["invalid_key"]

    def test_set_pre(self):
        pb = ParallelBlock({"a": PlusOne(), "b": PlusOne()})
        set_pre(pb, PlusOne())
        assert len(pb.pre) == 1

        block = Block(pb)
        assert not get_pre(Block())
        set_pre(block, PlusOne())
        assert len(get_pre(block)) == 1

    def test_input_schema_pre(self):
        pb = ParallelBlock({"a": PlusOne(), "b": PlusOne()})
        outputs = mm.schema.trace(pb, torch.randn(1, 3))
        input_schema = mm.schema.input(pb)
        assert len(input_schema) == 1
        assert len(mm.schema.output(pb)) == 2
        assert len(outputs) == 2

        pb2 = ParallelBlock({"a": PlusOne(), "b": PlusOne()})
        assert not get_pre(pb2)
        pb2.prepend(pb)
        assert not get_pre(pb2) == pb
        assert get_pre(pb2)[0] == pb
        pb2.append(pb)

        assert input_schema == mm.schema.input(pb2)
        assert mm.schema.output(pb2) == mm.schema.output(pb)


class TestResidualBlock:
    def test_forward(self):
        input_tensor = torch.randn(1, 3, 64, 64)
        conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        residual = ResidualBlock(conv)

        output_tensor = module_utils.module_test(residual, input_tensor)
        expected_tensor = input_tensor + conv(input_tensor)

        assert torch.allclose(output_tensor, expected_tensor)


class TestShortcutBlock:
    def test_forward(self):
        input_tensor = torch.randn(1, 3, 64, 64)
        conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        shortcut = ShortcutBlock(conv)

        output_dict = module_utils.module_test(shortcut, input_tensor)

        assert "output" in output_dict
        assert "shortcut" in output_dict
        assert torch.allclose(output_dict["output"], conv(input_tensor))
        assert torch.allclose(output_dict["shortcut"], input_tensor)
