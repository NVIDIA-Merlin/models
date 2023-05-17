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

    Parameters
    ----------
    loss: nn.Module
        The loss function used for training.
    metrics: Sequence[Metric]
        The metrics used for evaluation.
    name: Optional[str]
        The name of the model output.
    schema: Optional[ColumnSchema]
        The schema defining the column properties.
    """

    def __init__(
        self,
        *module: nn.Module,
        loss: nn.Module,
        metrics: Sequence[Metric] = (),
        name: Optional[str] = None,
        schema: Optional[ColumnSchema] = None,
    ):
        """Initializes a ModelOutput object."""
        super().__init__(*module, name=name)

        self.loss = loss
        self.metrics = metrics
        self._output_schema = None

        if schema:
            self.setup_schema(schema)
        self.register_buffer("target", torch.zeros(1, dtype=torch.float32))

    def setup_schema(self, schema: Optional[ColumnSchema]):
        """Set up the schema for the output.

        Parameters
        ----------
        schema: ColumnSchema or None
            The schema defining the column properties.
        """
        self._output_schema = Schema([schema])

    @property
    def output_schema(self):
        """The schema of the output.

        Returns
        -------
        Schema or None
            The schema defining the column properties.
        """
        return self._output_schema

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
