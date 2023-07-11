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
from typing import Dict, Optional, Tuple

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
        assert mm.output_schema(block) == mm.output_schema.tensors(inputs)

    def test_insertion(self):
        block = Block()
        block.prepend(PlusOne())
        block.append(PlusOne())

        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        outputs = module_utils.module_test(block, inputs, batch=Batch(inputs))

        assert torch.equal(outputs, inputs + 2)

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
        schema = mm.output_schema(pb)

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
        input_schema = mm.input_schema(pb)
        assert len(input_schema) == 1
        assert len(mm.output_schema(pb)) == 2
        assert len(outputs) == 2

        pb2 = ParallelBlock({"a": PlusOne(), "b": PlusOne()})
        assert not get_pre(pb2)
        pb2.prepend(pb)
        assert not get_pre(pb2) == pb
        assert get_pre(pb2)[0] == pb
        pb2.append(pb)

        assert input_schema == mm.input_schema(pb2)
        assert mm.output_schema(pb2) == mm.output_schema(pb)

    def test_leaf(self):
        block = ParallelBlock({"a": PlusOne()})

        assert isinstance(block.leaf(), PlusOne)

        block.branches["b"] = PlusOne()
        with pytest.raises(ValueError):
            block.leaf()

        block.prepend(PlusOne())
        with pytest.raises(ValueError):
            block.leaf()

        block = ParallelBlock({"a": nn.Sequential(PlusOne())})
        assert isinstance(block.leaf(), PlusOne)


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

    def test_nesting(self):
        inputs = torch.rand(5, 5)
        shortcut = ShortcutBlock(ShortcutBlock(PlusOne()))
        output = module_utils.module_test(shortcut, inputs)

        assert torch.equal(output["shortcut"], inputs)
        assert torch.equal(output["output"], inputs + 1)

    def test_convert(self):
        block = Block(PlusOne())
        shortcut = ShortcutBlock(*block)
        nested = ShortcutBlock(ShortcutBlock(shortcut), propagate_shortcut=True)

        assert isinstance(shortcut[0], PlusOne)
        inputs = torch.rand(5, 5)
        assert torch.equal(
            module_utils.module_test(shortcut, inputs)["output"],
            module_utils.module_test(nested, inputs)["output"],
        )

    def test_with_parallel(self):
        parallel = ParallelBlock({"a": PlusOne(), "b": PlusOne()})
        shortcut = ShortcutBlock(parallel)

        inputs = torch.rand(5, 5)

        outputs = shortcut(inputs)

        outputs = module_utils.module_test(shortcut, inputs)
        assert torch.equal(outputs["shortcut"], inputs)
        assert torch.equal(outputs["a"], inputs + 1)
        assert torch.equal(outputs["b"], inputs + 1)

    def test_propagate_shortcut(self):
        class PlusOneShortcut(nn.Module):
            def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
                return inputs["shortcut"] + 1

        shortcut = ShortcutBlock(PlusOneShortcut(), propagate_shortcut=True)
        shortcut = ShortcutBlock(shortcut, propagate_shortcut=True)
        inputs = torch.rand(5, 5)
        outputs = module_utils.module_test(shortcut, inputs)

        assert torch.equal(outputs["output"], inputs + 1)

        with pytest.raises(RuntimeError):
            shortcut({"a": inputs})

    def test_exception(self):
        with_tuple = Block(PlusOneTuple())
        shortcut = ShortcutBlock(with_tuple)

        with pytest.raises(RuntimeError):
            module_utils.module_test(shortcut, torch.rand(5, 5))

        class PlusOneShortcutTuple(nn.Module):
            def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
                return inputs["shortcut"] + 1, inputs["shortcut"]

        shortcut_propagate = ShortcutBlock(PlusOneShortcutTuple(), propagate_shortcut=True)
        with pytest.raises(RuntimeError):
            module_utils.module_test(shortcut_propagate, torch.rand(5, 5))


class TestBatchBlock:
    def test_forward_with_batch(self):
        batch = Batch(torch.tensor([1, 2]), torch.tensor([3, 4]))
        outputs = mm.BatchBlock()(batch)

        assert batch == outputs

    def test_forward_with_features(self):
        feat = torch.tensor([1, 2])
        outputs = module_utils.module_test(mm.BatchBlock(), feat)
        assert isinstance(outputs, mm.Batch)
        assert torch.equal(outputs.feature(), feat)

    def test_forward_with_tuple(self):
        feat, target = torch.tensor([1, 2]), torch.tensor([3, 4])
        outputs = module_utils.module_test(mm.BatchBlock(), feat, targets=target)

        assert isinstance(outputs, mm.Batch)
        assert torch.equal(outputs.feature(), feat)
        assert torch.equal(outputs.target(), target)

    def test_forward_exception(self):
        with pytest.raises(
            RuntimeError, match="Features must be a tensor or a dictionary of tensors"
        ):
            module_utils.module_test(mm.BatchBlock(), (torch.tensor([1, 2]), torch.tensor([1, 2])))

    def test_nested(self):
        feat, target = torch.tensor([1, 2]), torch.tensor([3, 4])
        outputs = module_utils.module_test(mm.BatchBlock(mm.BatchBlock()), feat, targets=target)

        assert isinstance(outputs, mm.Batch)
        assert torch.equal(outputs.feature(), feat)
        assert torch.equal(outputs.target(), target)

    def test_in_parallel(self):
        feat, target = torch.tensor([1, 2]), torch.tensor([3, 4])
        outputs = module_utils.module_test(
            mm.BatchBlock(mm.ParallelBlock({"a": mm.BatchBlock()})), feat, targets=target
        )

        assert isinstance(outputs, mm.Batch)
        assert torch.equal(outputs.feature(), feat)
        assert torch.equal(outputs.target(), target)

    def test_exception(self):
        class BatchToTuple(nn.Module):
            def forward(
                self, inputs: Dict[str, torch.Tensor], batch: Optional[Batch] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                return inputs["default"], inputs["default"]

        feat, target = torch.tensor([1, 2]), torch.tensor([3, 4])
        with pytest.raises(RuntimeError, match="Module must return a Batch"):
            module_utils.module_test(mm.BatchBlock(BatchToTuple()), feat, targets=target)
