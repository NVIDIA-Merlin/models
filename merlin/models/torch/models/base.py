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
from functools import reduce
from typing import Dict, List, Optional, Union

import torch
from pytorch_lightning import LightningModule
from torch import nn

from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.batch import Batch, sample_batch
from merlin.models.torch.block import Block
from merlin.models.torch.container import BlockContainer
from merlin.models.torch.outputs.base import ModelOutput
from merlin.models.torch.utils import module_utils
from merlin.models.utils.registry import camelcase_to_snakecase
from merlin.schema import Schema


class Model(Block, LightningModule):
    """Merlin Model class"""

    def __init__(
        self,
        *blocks: nn.Module,
        schema: Optional[Schema] = None,
        optimizer=torch.optim.Adam,
    ):
        """Initializes `Model` class"""
        super().__init__()
        self.schema = schema

        self.pre = BlockContainer(name="pre")
        self.blocks = BlockContainer(name="blocks")
        for block in blocks:
            self.blocks.append(block)
        self.post = BlockContainer(name="post")

        self.optimizer = optimizer

    def initialize(self, data: Union[Dataset, Loader, Batch]):
        return initialize(self, data)

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        outputs = inputs
        for pre in self.pre.values:
            outputs = pre(outputs, batch=batch)
        for block in self.blocks.values:
            outputs = block(outputs, batch=batch)
        for post in self.post.values:
            outputs = post(outputs, batch=batch)
        return outputs

    def training_step(self, batch, batch_idx):
        del batch_idx
        inputs, targets = batch
        predictions = self(inputs)

        loss = compute_loss(predictions, targets, self.model_outputs())
        for name, value in loss.items():
            self.log(f"train_{name}", value)

        return loss["loss"]

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def model_outputs(self) -> List[ModelOutput]:
        return module_utils.find_all_instances(self, ModelOutput)

    def first(self) -> nn.Module:
        return self.blocks.values[0]

    def last(self) -> nn.Module:
        return self.blocks.values[-1]

    def input_schema(self) -> Schema:
        if self.schema:
            return self.schema
        return Schema([])

    def output_schema(self) -> Schema:
        output_schemas = []
        for child in module_utils.get_all_children(self):
            if hasattr(child, "output_schema"):
                output_schemas.append(child.output_schema())

        if not output_schemas:
            raise ValueError("No output schema found")

        return reduce(lambda a, b: a + b, output_schemas)


def initialize(module, data: Union[Dataset, Loader, Batch]):
    if isinstance(data, (Loader, Dataset)):
        module.double()  # TODO: Put in data-loader PR to standardize on float-32
        batch = sample_batch(data, batch_size=1, shuffle=False)
    elif isinstance(data, Batch):
        batch = data
    else:
        raise RuntimeError(f"Unexpected input type: {type(data)}")

    module.to(batch.device())
    return module(batch.features)


def compute_loss(
    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
    targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    model_outputs: List[ModelOutput],
    compute_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    """ """
    if len(model_outputs) < 1:
        raise RuntimeError("No model outputs found.")
    if (
        isinstance(predictions, torch.Tensor)
        and isinstance(targets, torch.Tensor)
        and len(model_outputs) > 1
    ):
        raise RuntimeError("Multiple outputs but only one target was provided.")

    results = {"loss": torch.tensor(0.0)}
    for model_out in model_outputs:
        name = model_out.output_schema.first.name

        if targets is None:
            _targets = model_out.target
        elif isinstance(targets, torch.Tensor):
            _targets = targets
        elif isinstance(targets, dict):
            _targets = targets[name]
        # _targets = torch.squeeze(_targets).double()

        if isinstance(predictions, torch.Tensor):
            _predictions = predictions
        elif isinstance(predictions, dict):
            _predictions = predictions[name]
        # _predictions = torch.squeeze(_predictions)

        # TODO: How to handle task weights?
        results["loss"] += model_out.loss(_predictions, _targets) / len(model_outputs)

        if not compute_metrics:
            continue

        for metric in model_out.metrics:
            metric_name = camelcase_to_snakecase(metric.__class__.__name__)
            results[metric_name] = metric(_predictions, _targets)

    return results
