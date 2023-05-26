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

from torch import nn
from torchmetrics import MeanSquaredError, Metric

import merlin.dtypes as md
from merlin.models.torch.outputs.base import ModelOutput
from merlin.schema import ColumnSchema, Schema, Tags


class RegressionOutput(ModelOutput):
    """A prediction block for regression.

    Parameters
    ----------
    schema: Optional[ColumnSchema]
        The schema defining the column properties. Default is None.
    loss: torch.nn.Module
        The loss function used for training. Default is nn.MSELoss().
    metrics: Sequence[Metric]
        The metrics used for evaluation. Default is MeanSquaredError.
    """

    def __init__(
        self,
        schema: Optional[ColumnSchema] = None,
        loss: nn.Module = nn.MSELoss(),
        metrics: Sequence[Metric] = (MeanSquaredError(),),
    ):
        """Initializes a RegressionOutput object."""
        super().__init__(
            nn.LazyLinear(1),
            schema=schema,
            loss=loss,
            metrics=metrics,
        )

    def setup_schema(self, target: Optional[ColumnSchema]):
        """Set up the schema for the output.

        Parameters
        ----------
        target: Optional[ColumnSchema]
            The schema defining the column properties.
        """
        _target = target.with_dtype(md.float32).with_tags([Tags.CONTINUOUS])

        self.output_schema = Schema([_target])
