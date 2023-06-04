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
from typing import Dict, List, Optional, Sequence, Union

import torch
from pytorch_lightning import LightningModule
from torch import nn

from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.batch import Batch
from merlin.models.torch.block import Block
from merlin.models.torch.container import BlockContainer
from merlin.models.torch.outputs.base import ModelOutput
from merlin.models.torch.utils import module_utils
from merlin.models.utils.registry import camelcase_to_snakecase
from merlin.schema import Schema


class Model(Block, LightningModule):
    """
    Merlin Model class.

    The Model class extends from both the Block and LightningModule classes. It
    allows for easy construction of models using pre-defined blocks.

    Parameters
    ----------
    *blocks: nn.Module
        One or more blocks that make up the core functionality of the model.
    schema: Schema, optional
        A Merlin schema. Default is None.
    optimizer: torch.optim.Optimizer, optional
        A PyTorch optimizer from the PyTorch library (or any custom optimizer
        that follows the same API). Default is Adam optimizer.

    Example usage
    -------------
    >>> model = Model(
    ...    TabularInputBlock(schema),
    ...    MLPBlock([32, 16]),
    ...    BinaryOutput(schema.select_by_tag(Tags.TARGET).first),
    ... )
    ... trainer = Trainer(max_epochs=1)
    ... with Loader(dataset, batch_size=16) as loader:
    ...     model.initialize(loader)
    ...     trainer.fit(model, loader)
    """

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
        """Initializes the model based on a given data set."""
        return module_utils.initialize(self, data)

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        """Performs a forward pass through the model."""
        outputs = inputs
        for pre in self.pre.values:
            outputs = pre(outputs, batch=batch)
        for block in self.blocks.values:
            outputs = block(outputs, batch=batch)
        for post in self.post.values:
            outputs = post(outputs, batch=batch)
        return outputs

    def training_step(self, batch, batch_idx):
        """Performs a training step with a single batch."""
        del batch_idx
        if isinstance(batch, Batch):
            features = batch.features
            targets = batch.targets
        else:
            features, targets = batch

        predictions = self(features, batch=Batch(features, targets))

        loss_and_metrics = compute_loss(predictions, targets, self.model_outputs())
        for name, value in loss_and_metrics.items():
            self.log(f"train_{name}", value)

        return loss_and_metrics["loss"]

    def configure_optimizers(self):
        """Configures the optimizer for the model."""
        return self.optimizer(self.parameters())

    def model_outputs(self) -> List[ModelOutput]:
        """Finds all instances of `ModelOutput` in the model."""
        return module_utils.find_all_instances(self, ModelOutput)

    def first(self) -> nn.Module:
        """Returns the first block in the model."""
        return self.blocks.values[0]

    def last(self) -> nn.Module:
        """Returns the last block in the model."""
        return self.blocks.values[-1]

    def input_schema(self) -> Schema:
        """Returns the input schema of the model."""
        if self.schema:
            return self.schema
        # TODO: Implement logic when TabularInputBlock is available.
        return Schema([])

    def output_schema(self) -> Schema:
        output_schemas = []
        for child in module_utils.get_all_children(self):
            if hasattr(child, "output_schema"):
                output_schemas.append(child.output_schema())

        if not output_schemas:
            raise RuntimeError("No output schema found")

        return reduce(lambda a, b: a + b, output_schemas)  # type: ignore


def compute_loss(
    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
    targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]],
    model_outputs: Sequence[ModelOutput],
    compute_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute the loss and metrics for the given model outputs.

    This function takes in predictions and targets, and a list of model
    outputs. It computes the loss using the loss function of each model output
    and averages it. If `compute_metrics` is set to True, it also computes the
    metrics defined in each model output.

    Parameters
    ----------
    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]]
        The predictions from the model.
    targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]
        The ground truth targets.
    model_outputs: Sequence[ModelOutput]
        A list of model outputs. Each model output must have a defined loss
        function.
    compute_metrics: bool, optional
        Whether to compute metrics defined in each model output. Default: True.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary containing the loss and the computed metrics (if any).

    Raises
    ------
    RuntimeError: If no model outputs are provided, or if multiple model
        outputs are provided but only one set of targets is given.

    Example usage
    -------------
    >>> predictions = torch.tensor([0.2, 0.3, 0.6, 0.8])
    >>> targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    >>> binary_output = mm.BinaryOutput(ColumnSchema("target"))
    >>> results = compute_loss(predictions, targets, [binary_output])
    >>> results["loss"]
    tensor(0.7653)
    >>> results["binary_accuracy"]
    tensor(0.5000)
    """
    if len(model_outputs) < 1:
        raise RuntimeError("No model outputs found.")

    results = {"loss": torch.tensor(0.0)}
    for model_out in model_outputs:
        name = model_out.output_schema.first.name

        if targets is None or (isinstance(targets, dict) and name not in targets):
            if not hasattr(model_out, "target"):
                raise ValueError(f"'{model_out.__class__.__name__}' has no target.")
            if isinstance(predictions, dict):
                pred_col = predictions[name]
            else:
                pred_col = predictions
            _targets = torch.ones_like(pred_col) * model_out.target
        elif isinstance(targets, dict):
            _targets = targets[name]
        elif isinstance(targets, torch.Tensor):
            _targets = targets
        else:
            raise ValueError(f"Unknown 'targets' type: {type(targets)}")

        if isinstance(predictions, dict):
            if name not in predictions:
                raise RuntimeError(f"Column '{name}' not found in predictions")
            _predictions = predictions[name]
        elif isinstance(predictions, torch.Tensor):
            _predictions = predictions
        else:
            raise ValueError(f"Unknown 'predictions' type: {type(predictions)}")

        results["loss"] = results["loss"] + model_out.loss(_predictions, _targets) / len(
            model_outputs
        )

        if not compute_metrics:
            continue

        for metric in model_out.metrics:
            metric_name = camelcase_to_snakecase(metric.__class__.__name__)
            results[metric_name] = metric(_predictions, _targets)
    return results
