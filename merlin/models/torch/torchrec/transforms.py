from typing import Dict, Union

import torch
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

from merlin.models.torch.base import registry
from merlin.models.torch.transforms.aggregation import ConcatFeatures
from merlin.schema import Schema


class ToKeyedJaggedTensor(nn.Module):
    """Convert inputs to a KeyedJaggedTensor."""

    def __init__(self, schema: Schema):
        super().__init__()
        self.schema = schema

    def forward(self, inputs) -> KeyedJaggedTensor:
        if isinstance(inputs, KeyedJaggedTensor):
            return inputs

        jagged_dict = {}

        for col in self.schema:
            if col.is_ragged or col.is_list:
                jagged_dict[col.name] = JaggedTensor(
                    inputs[col.name + "__values"], offsets=inputs[col.name + "__offsets"]
                )
            else:
                if col.name not in inputs:
                    continue
                jagged_dict[col.name] = JaggedTensor(
                    inputs[col.name], lengths=torch.ones_like(inputs[col.name], dtype=torch.int)
                )

        output = KeyedJaggedTensor.from_jt_dict(jagged_dict)

        return output


@registry.register("to-dict")
class ToDict(nn.Module):
    """Convert inputs to a dictionary."""

    def forward(self, inputs) -> Dict[str, Union[torch.Tensor, JaggedTensor]]:
        if hasattr(inputs, "to_dict"):
            return inputs.to_dict()

        if isinstance(inputs, torch.Tensor):
            return {"tensor": inputs}

        return inputs


class ToTorchRecBatch(nn.Module):
    def __init__(self, schema: Schema, dense_concat=ConcatFeatures()):
        super().__init__()
        self.schema = schema
        self.dense_concat = dense_concat

    def forward(self, inputs, targets=None) -> Batch:
        dense_dict = {}
        jagged_dict = {}

        for col in self.schema:
            if col.is_ragged or col.is_list:
                values = inputs.get(col.name + "__values", None)
                offsets = inputs.get(col.name + "__offsets", None)

                if not values or not offsets:
                    continue

                jagged_dict[col.name] = JaggedTensor(values, offsets=offsets)
            else:
                if col.name not in inputs:
                    continue
                dense_dict[col.name] = inputs[col.name]

        dense = self.dense_concat(dense_dict)
        key_jagged = KeyedJaggedTensor.from_jt_dict(jagged_dict)

        return Batch(dense_features=dense, sparse_features=key_jagged, targets=targets)
