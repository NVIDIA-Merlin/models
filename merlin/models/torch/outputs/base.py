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
import inspect
from copy import deepcopy
from typing import Optional, Sequence

import torch
from torch import nn
from torchmetrics import Metric

from merlin.models.torch.block import Block
from merlin.models.torch.transforms.bias import LogitsTemperatureScaler


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
    loss: nn.Module
        The loss function used for training.
    metrics: Sequence[Metric]
        The metrics used for evaluation.
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.0
    name: Optional[str]
        The name of the model output.
    """

    def __init__(
        self,
        *module: nn.Module,
        loss: Optional[nn.Module] = None,
        metrics: Optional[Sequence[Metric]] = None,
        logits_temperature: float = 1.0,
        name: Optional[str] = None,
    ):
        """Initializes a ModelOutput object."""
        super().__init__(*module, name=name)

        self.loss = loss
        self.metrics = metrics

        self.create_target_buffer()
        if logits_temperature != 1.0:
            self.append(LogitsTemperatureScaler(logits_temperature))

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

    def copy(self):
        metrics = deepcopy(self.metrics)
        self.metrics = []

        output = deepcopy(self)

        copied_metrics = []
        for metric in metrics:
            params = inspect.signature(metric.__class__.__init__).parameters
            kwargs = {}
            for arg_name, arg_value in params.items():
                if arg_name in metric.__dict__:
                    kwargs[arg_name] = metric.__dict__[arg_name]
            m = metric.__class__(**kwargs)
            m.load_state_dict(metric.state_dict())
            copied_metrics.append(m)

        self.metrics = metrics
        output.metrics = copied_metrics
        output.loss = deepcopy(self.loss)

        return output
