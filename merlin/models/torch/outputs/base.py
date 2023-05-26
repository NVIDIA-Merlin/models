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
from typing import Optional, Sequence

import torch
from torch import nn
from torchmetrics import Metric

from merlin.models.torch.block import Block
from merlin.schema import ColumnSchema, Schema


class ModelOutput(Block):
    """A base class for prediction tasks.

    Example usage::
        >>> schema = ColumnSchema(
        ...    "target",
        ...    properties={"domain": {"min": 0, "max": 1}},
        ...    tags=[Tags.CATEGORICAL, Tags.TARGET]
        ... )
        >>> model_output = ModelOutput(
        ...    nn.LazyLinear(1),
        ...    nn.Sigmoid(),
        ...    schema=schema
        ... )
        >>> input = torch.randn(3, 2)
        >>> output = model_output(input)
        >>> print(output)
        tensor([[0.5529],
                [0.3562],
                [0.7478]], grad_fn=<SigmoidBackward0>)

    Parameters
    ----------
    schema: Optional[ColumnSchema]
        The schema defining the column properties.
    loss: nn.Module
        The loss function used for training.
    metrics: Sequence[Metric]
        The metrics used for evaluation.
    name: Optional[str]
        The name of the model output.
    """

    def __init__(
        self,
        *module: nn.Module,
        schema: Optional[ColumnSchema] = None,
        loss: Optional[nn.Module] = None,
        metrics: Sequence[Metric] = (),
        name: Optional[str] = None,
    ):
        """Initializes a ModelOutput object."""
        super().__init__(*module, name=name)

        self.loss = loss
        self.metrics = metrics
        self.output_schema: Schema = Schema()

        if schema:
            self.setup_schema(schema)
        self.create_target_buffer()

    def setup_schema(self, schema: Optional[ColumnSchema]):
        """Set up the schema for the output.

        Parameters
        ----------
        schema: ColumnSchema or None
            The schema defining the column properties.
        """
        self.output_schema = Schema([schema])

    def create_target_buffer(self):
        self.register_buffer("target", torch.zeros(1, dtype=torch.float32))

    def eval(self):
        """Sets the module in evaluation mode.

        Returns
        -------
        nn.Module
            The module in evaluation mode.
        """
        # Reset target
        self.target = torch.zeros(1, dtype=torch.float32)

        return self.train(False)
