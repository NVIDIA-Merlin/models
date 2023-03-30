from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from merlin.io import Dataset
from merlin.models.torch.combinators import SequentialBlock
from merlin.models.torch.inputs.base import TabularInputBlock
from merlin.models.torch.predict import batch_predict
from merlin.models.torch.transforms.aggregation import ConcatFeatures
from merlin.models.torch.utils.module_utils import module_name
from merlin.schema import ColumnSchema, Schema, Tags


class Encoder(SequentialBlock):
    """An Encoder encodes features into a latent space."""

    def __init__(
        self,
        inputs: Union[Schema, nn.Module],
        *blocks: nn.Module,
        pre=None,
        post=None,
        output_name: Optional[str] = None,
    ):
        if isinstance(inputs, Schema):
            input_block = TabularInputBlock(inputs)
        else:
            input_block = inputs

        if not hasattr(input_block, "input_schema"):
            raise ValueError("First block must have an input_schema")

        super().__init__(input_block, *blocks, pre=pre, post=post)

        if not output_name:
            output_name = module_name(self)

        self.output_name = output_name
        self.concat = ConcatFeatures()

    def forward(self, inputs) -> torch.Tensor:
        output = super().forward(inputs)

        if isinstance(output, dict):
            output = self.concat(output)

        if not isinstance(output, torch.Tensor):
            raise ValueError("Encoder output must be a tensor.")

        # We record the shape of the output to know the output-schema
        self.register_buffer("output_shape", torch.tensor(output.shape))

        return output

    def encode(
        self,
        dataset: Dataset,
        batch_size: int,
        index: Union[str, ColumnSchema, Schema, Tags],
        **kwargs,
    ) -> Dataset:
        return self.batch_predict(
            dataset,
            batch_size=batch_size,
            add_inputs=False,
            index=index,
            **kwargs,
        )

    def batch_predict(
        self,
        dataset: Dataset,
        batch_size: int,
        index: Optional[Union[str, ColumnSchema, Schema, Tags]] = None,
        add_inputs: bool = True,
        **kwargs,
    ) -> Dataset:
        _index = self._parse_index(index).first.name if index else None

        return batch_predict(
            self,
            self.encoder_output_schema,
            dataset,
            batch_size,
            index=_index,
            add_inputs=add_inputs,
            **kwargs,
        )

    def _parse_index(self, index) -> Schema:
        if isinstance(index, ColumnSchema):
            index = Schema([index])
        elif isinstance(index, str):
            index = Schema([self.schema[index]])
        elif isinstance(index, Tags):
            index = self.schema.select_by_tag(index)
        elif not isinstance(index, Schema):
            raise ValueError(f"Invalid index: {index}")

        return index

    @property
    def input_schema(self) -> Schema:
        return self[0].input_schema

    @property
    def encoder_output_schema(self) -> Schema:
        if not hasattr(self, "output_shape"):
            raise ValueError("Encoder has not been called yet.")

        dims = (None,) + tuple(self.output_shape.cpu().numpy().tolist())[1:]
        col = ColumnSchema(self.output_name, dims=dims, dtype=np.float32)

        return Schema([col])
