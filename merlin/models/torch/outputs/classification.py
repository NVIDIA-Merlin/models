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
from typing import Optional, Sequence, Union

from torch import nn
from torchmetrics import AUROC, Accuracy, Metric, Precision, Recall

import merlin.dtypes as md
from merlin.models.torch.outputs.base import ModelOutput
from merlin.schema import ColumnSchema, Schema


class BinaryOutput(ModelOutput):
    """A prediction block for binary classification.

    Parameters
    ----------
    schema: Optional[ColumnSchema])
        The schema defining the column properties. Default is None.
    loss: nn.Module
        The loss function used for training. Default is nn.BCEWithLogitsLoss().
    metrics: Sequence[Metric]
        The metrics used for evaluation. Default includes Accuracy, AUROC, Precision, and Recall.
    """

    DEFAULT_LOSS_CLS = nn.BCEWithLogitsLoss
    DEFAULT_METRICS_CLS = (Accuracy, AUROC, Precision, Recall)

    def __init__(
        self,
        schema: Optional[ColumnSchema] = None,
        loss: Optional[nn.Module] = None,
        metrics: Sequence[Metric] = (),
    ):
        """Initializes a BinaryOutput object."""
        super().__init__(
            nn.LazyLinear(1),
            nn.Sigmoid(),
            schema=schema,
            loss=loss or self.DEFAULT_LOSS_CLS(),
            metrics=metrics or [m(task="binary") for m in self.DEFAULT_METRICS_CLS],
        )

    def setup_schema(self, target: Optional[Union[ColumnSchema, Schema]]):
        """Set up the schema for the output.

        Parameters
        ----------
        target: Optional[ColumnSchema]
            The schema defining the column properties.
        """
        if isinstance(target, Schema):
            if len(target) != 1:
                raise ValueError("Schema must contain exactly one column.")

            target = target.first

        _target = target.with_dtype(md.float32)
        if "domain" not in target.properties:
            _target = _target.with_properties(
                {"domain": {"min": 0, "max": 1, "name": _target.name}},
            )

        self.output_schema = Schema([_target])
