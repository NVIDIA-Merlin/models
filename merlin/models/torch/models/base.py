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
import os
from typing import Dict, List, Optional, Sequence, Union

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch import nn

from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.batch import Batch
from merlin.models.torch.block import Block
from merlin.models.torch.outputs.base import ModelOutput
from merlin.models.torch.utils import module_utils
from merlin.models.utils.registry import camelcase_to_snakecase


class Model(LightningModule, Block):
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
    ... trainer.fit(model, Loader(dataset, batch_size=16))
    """

    def __init__(self, *blocks: nn.Module, optimizer=torch.optim.Adam, initialization="auto"):
        super().__init__()

        # Copied from BlockContainer.__init__
        self.values = nn.ModuleList()
        for module in blocks:
            self.values.append(self.wrap_module(module))

        self.optimizer = optimizer
        self.initialization = initialization

    def initialize(self, data: Union[Dataset, Loader, Batch]):
        """Initializes the model based on a given data set."""
        return module_utils.initialize(self, data, dtype=self._dtype)

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        """Performs a forward pass through the model."""
        outputs = inputs
        for block in self.values:
            outputs = block(outputs, batch=batch)
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

        # TODO: Fix compute_metrics=True for multi-gpu
        loss_and_metrics = compute_loss(
            predictions, targets, self.model_outputs(), compute_metrics=True
        )
        for name, value in loss_and_metrics.items():
            self.log(f"train_{name}", value)

        return loss_and_metrics["loss"]

    def validation_step(self, batch, batch_idx):
        return self._val_step(batch, batch_idx, type="val")

    def test_step(self, batch, batch_idx):
        return self._val_step(batch, batch_idx, type="test")

    def _val_step(self, batch, batch_idx, type="val"):
        del batch_idx
        if not isinstance(batch, Batch):
            batch = Batch(features=batch[0], targets=batch[1])

        predictions = self(batch.features, batch=batch)

        loss_and_metrics = compute_loss(predictions, batch.targets, self.model_outputs())
        for name, value in loss_and_metrics.items():
            self.log(f"{type}_{name}", value)

        return loss_and_metrics

    def configure_optimizers(self):
        """Configures the optimizer for the model."""
        return self.optimizer(self.parameters())

    def model_outputs(self) -> List[ModelOutput]:
        """Finds all instances of `ModelOutput` in the model."""
        return self.find(ModelOutput)

    def first(self) -> nn.Module:
        """Returns the first block in the model."""
        return self.values[0]

    def last(self) -> nn.Module:
        """Returns the last block in the model."""
        return self.values[-1]

    def setup(self, stage):
        """Initialize the model if `initialization="auto"`."""
        if self.initialization == "auto":
            loop = getattr(self.trainer, f"{stage}_loop")

            data_instance = loop._data_source.instance
            if isinstance(data_instance, MultiLoader):
                self.initialize(data_instance.batch.to(None, device=self.device))
            else:
                dataloader = loop._data_source.dataloader()
                if isinstance(dataloader, Loader):
                    self.initialize(dataloader)
                else:
                    raise ValueError(
                        f"Can't auto-initialize from a non-merlin dataloader, got: {dataloader}",
                        "Please initialize the model manually with `model.initialize(batch)`",
                    )

    def teardown(self, stage: str) -> None:
        """Teardown the data-loader after training."""
        loop = getattr(self.trainer, f"{stage}_loop")
        dataloader = loop._data_source.dataloader()
        if isinstance(dataloader, Loader):
            dataloader.stop()


class MultiLoader(LightningDataModule):
    """
    Data Module for handling multiple types of data loaders. It facilitates the usage
    of multiple datasets, as well as distributed training on multiple GPUs.

    This class is particularly useful in scenarios where you have separate train,
    validation and test datasets, and you want to use PyTorch Lightning's Trainer
    which requires a single DataModule.

    Parameters
    ----------
    train : Union[Dataset, Loader]
        Training dataset or data loader.
    valid : Optional[Union[Dataset, Loader]], optional
        Validation dataset or data loader, by default None
    test : Optional[Union[Dataset, Loader]], optional
        Test dataset or data loader, by default None
    repartition : int, optional
        Number of partitions to divide the dataset into, by default None
    batch_size : int, optional
        Number of data points per batch, by default 1024


    Example usage for multi-GPU::
        model = mm.Model(...)
        train, valid = generate_data(...)
        model.initialize(train)

        trainer = pl.Trainer(max_epochs=5, devices=[0, 1])
        trainer.fit(model, mm.MultiLoader(train, valid, batch_size=1024, repartition=4))
    """

    def __init__(
        self,
        train: Union[Dataset, Loader],
        valid: Optional[Union[Dataset, Loader]] = None,
        test: Optional[Union[Dataset, Loader]] = None,
        batch_size: int = 1024,
        repartition: Optional[int] = None,
    ):
        super().__init__()
        self.repartition = repartition
        self.train = train
        self.batch_size = batch_size
        self.batch = Batch.sample_from(train, batch_size=1, shuffle=False)
        if valid:
            self.val_dataloader = lambda: self._create_loader(valid, "valid")
        if test:
            self.test_dataloader = lambda: self._create_loader(test, "test")

    def train_dataloader(self) -> Loader:
        return self._create_loader(self.train, "train")

    def _create_loader(self, data: Union[Dataset, Loader], name: str) -> Loader:
        """
        Create a data loader with the right arguments.

        Parameters
        ----------
        data : Union[Dataset, Loader]
            The input data, can be a dataset or data loader.
        name : str
            Name of the data loader.

        Returns
        -------
        Loader
            The created data loader.
        """

        _dataset = data.dataset if isinstance(data, Loader) else data

        if self.repartition:
            _dataset = _dataset.repartition(self.repartition)

        has_world_size = "WORLD_SIZE" in os.environ
        if isinstance(data, Loader):
            output = Loader(
                _dataset,
                batch_size=data.batch_size,
                shuffle=data.shuffle,
                drop_last=int(os.environ["WORLD_SIZE"]) > 1 if has_world_size else data.drop_last,
                global_size=int(os.environ["WORLD_SIZE"]) if has_world_size else data.global_size,
                global_rank=int(os.environ["LOCAL_RANK"]) if has_world_size else data.global_rank,
                transforms=data.transforms,
            )
        else:
            output = Loader(
                _dataset,
                batch_size=self.batch_size,
                drop_last=int(os.environ["WORLD_SIZE"]) > 1 if has_world_size else False,
                global_size=int(os.environ["WORLD_SIZE"]) if has_world_size else None,
                global_rank=int(os.environ["LOCAL_RANK"]) if has_world_size else None,
            )

        setattr(self, f"loader_{name}", output)
        return output

    def teardown(self, stage):
        """
        Stop all data loaders.
        """
        for attr in dir(self):
            if attr.startswith("loader"):
                if hasattr(getattr(self, attr), "stop"):
                    getattr(self, attr).stop()
                delattr(self, attr)


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

        if _targets.size() != _predictions.size():
            _targets = _targets.view(_predictions.size())
        if _targets.type() != _predictions.type():
            _targets = _targets.type_as(_predictions)

        results["loss"] = results["loss"] + model_out.loss(_predictions, _targets) / len(
            model_outputs
        )

        if not compute_metrics:
            continue

        for metric in model_out.metrics:
            metric_name = camelcase_to_snakecase(metric.__class__.__name__)
            if not metric.device or metric.device != _predictions.device:
                metric = metric.to(_predictions.device)

            results[metric_name] = metric(_predictions, _targets)
    return results
