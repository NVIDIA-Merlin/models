from typing import Dict, Optional

import torch
from torch import nn

from merlin.models.torch.data import sample_batch
from merlin.models.torch.testing import (
    BinaryOutput,
    Block,
    Model,
    TabularBatch,
    TabularBlock,
    TabularPadding,
    TabularSequence,
)
from merlin.models.torch.utils import module_utils
from merlin.schema import Tags


class WithBatch(nn.Module):
    def forward(self, x, batch: Optional[TabularBatch]):
        return x * 2


class TransformBatch(nn.Module):
    def forward(self, x, batch: TabularBatch) -> TabularBatch:
        return batch.replace(features={key: val * 2 for key, val in batch.features.items()})


class WithoutBatch(nn.Module):
    def forward(self, x):
        return x * 3


class WithoutBatchDict(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]):
        return {key: val * 3 for key, val in x.items()}


class TabularIdentity(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return x


class ConcatDict(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(x.values()), dim=-1)


class TransformBatchDict(nn.Module):
    def forward(self, x: Dict[str, torch.Tensor], batch: TabularBatch) -> TabularBatch:
        return batch.replace(features={key: val * 2 for key, val in batch.features.items()})


class TestBlockMixin:
    def test_register_block_hooks(self):
        pre_module = WithBatch()
        post_module = WithoutBatch()
        block = Block(nn.Identity(), pre=pre_module, post=post_module)

        assert block.pre.to_call == pre_module
        assert block.post.to_call == post_module
        assert block.pre.needs_batch is True
        assert block.post.needs_batch is False

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output = module_utils.module_test(block, input_tensor)

        assert torch.allclose(output, input_tensor * 3 * 2)

    def test_tabular(self):
        block = TabularBlock(TabularIdentity(), pre=WithoutBatchDict(), post=WithoutBatchDict())

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        inputs = {"a": input_tensor, "b": input_tensor}
        output = module_utils.module_test(block, inputs)

        for val in output.values():
            assert torch.allclose(val, input_tensor * 9)

    def test_tabular_with_agg(self):
        block = TabularBlock(
            TabularIdentity(), pre=WithoutBatchDict(), post=WithoutBatchDict(), agg=ConcatDict()
        )

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        inputs = {"a": input_tensor, "b": input_tensor}
        output = module_utils.module_test(block, inputs)

        assert isinstance(output, torch.Tensor)

    def test_sequential_module(self):
        block = Block(nn.Sequential(WithoutBatch(), WithBatch()))

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        output = module_utils.module_test(block, input_tensor)
        assert torch.allclose(output, input_tensor * 3 * 2)

    def test_batch_transform(self):
        block = Block(nn.Sequential(WithoutBatch(), TransformBatch()))
        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        batch = TabularBatch({"a": input_tensor, "b": input_tensor})

        output = module_utils.module_test(block, input_tensor, batch=batch)
        assert isinstance(output, TabularBatch)

        for val in output.features.values():
            assert torch.allclose(val, input_tensor * 2)


class TestBinaryOutput:
    def test_binary_output(self):
        inputs = torch.randn(10, 5)
        model_out = BinaryOutput()

        batch = TabularBatch(features={"a": inputs}, targets={"target": torch.ones(10, 1)})

        outputs = model_out(inputs, batch=batch)
        assert torch.allclose(model_out.target, batch.targets["target"])
        assert outputs.shape == (10, 1)

        outputs = module_utils.module_test(model_out, inputs, batch=batch)

        a = 5


class TestModel:
    def test_batch_transform_pre(self):
        model = Model(TabularIdentity(), pre=TransformBatchDict())

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        inputs = {"a": input_tensor, "b": input_tensor}
        batch = TabularBatch(features=inputs)

        output = module_utils.module_test(model, inputs, batch=batch)

        for val in output.values():
            assert torch.allclose(val, input_tensor * 2)


def test_padding(sequence_testing_data):
    features, targets = sample_batch(sequence_testing_data, 5)

    seq_schema = sequence_testing_data.schema.select_by_tag(Tags.SEQUENCE)
    padding = TabularPadding(seq_schema, seq_schema)

    batch = TabularBatch(features, targets)
    out = module_utils.module_test(padding, batch)
    # out = padding(TabularBatch(features, targets))

    assert isinstance(out, TabularBatch)
    assert isinstance(out.sequences, TabularSequence)
