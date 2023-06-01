import pytest
import torch
from torch import nn

from merlin.models.torch.utils.module_utils import module_test
from merlin.models.torch.utils.schema_utils import SchemaTrackingMixin
from merlin.schema import Schema, Tags


class TrackedModule(SchemaTrackingMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(10)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class TrackedDictModule(SchemaTrackingMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(10)

    def forward(self, x: torch.Tensor):
        return {"a": self.linear(x), "b": self.linear(x)}


class TestSchemaTrackingMixin:
    def test_tensor(self):
        inputs = torch.randn(1, 5)
        tracked_module = TrackedModule()
        module_test(tracked_module, inputs)

        schema = tracked_module.output_schema()
        assert isinstance(schema, Schema)
        assert len(schema) == 1
        assert len(schema.select_by_tag(Tags.EMBEDDING)) == 1

    def test_dict(self):
        inputs = torch.randn(1, 5)
        tracked_module = TrackedDictModule()

        outputs = tracked_module(inputs)
        traced_outputs = module_test(tracked_module, inputs)
        assert torch.equal(outputs["a"], traced_outputs["a"])
        assert torch.equal(outputs["b"], traced_outputs["b"])

        schema = tracked_module.output_schema()
        assert isinstance(schema, Schema)
        assert len(schema) == 2
        assert len(schema.select_by_tag(Tags.EMBEDDING)) == 2

    def test_exception(self):
        tracked_module = TrackedModule()
        with pytest.raises(RuntimeError):
            tracked_module.output_schema()

    def test_train(self):
        tracked_module = TrackedModule()
        tracked_module(torch.randn(1, 5))

        tracked_module.train()
        assert not tracked_module._forward_called

    def test_eval(self):
        tracked_module = TrackedModule()
        tracked_module(torch.randn(1, 5))

        tracked_module.eval()
        assert not tracked_module._forward_called
