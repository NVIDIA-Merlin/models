from functools import reduce
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch import nn

from merlin.dataloader.torch import Loader
from merlin.models.torch.base import register_post_hook, register_pre_hook
from merlin.models.torch.data import (
    needs_data_propagation_hook,
    register_data_propagation_hook,
    sample_batch,
)
from merlin.models.torch.outputs.base import ModelOutput
from merlin.models.torch.utils import module_utils
from merlin.schema import Schema


class Model(pl.LightningModule):
    def __init__(
        self,
        *blocks: nn.Module,
        pre=None,
        post=None,
        schema: Optional[Schema] = None,
        optimizer_cls=torch.optim.Adam,
    ):
        super().__init__()
        self.schema = schema
        self.blocks = nn.ModuleList(blocks)
        self.optimizer_cls = optimizer_cls

        self.pre = register_pre_hook(self, pre) if pre else None
        self.post = register_post_hook(self, post) if post else None
        self.data_propagation_hook = None
        self.testing = False
        if needs_data_propagation_hook(self):
            self.data_propagation_hook = register_data_propagation_hook(self)

    def forward(self, inputs):
        return module_utils.apply(self.blocks, inputs)

    def training_step(self, batch, batch_idx):
        del batch_idx
        inputs, targets = batch

        if self.data_propagation_hook:
            outputs = self(inputs, targets=targets)
        else:
            outputs = self(inputs)

        loss = self.calculate_loss(outputs, targets)
        self.log("train_loss", loss)

        return loss

    def calculate_loss(self, outputs, targets) -> torch.Tensor:
        model_output_dict = self.model_outputs_by_name

        if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
            assert len(model_output_dict) == 1, "Multiple outputs but only one target"

            model_output = list(model_output_dict.values())[0]

            return model_output.default_loss(outputs, targets)

        loss = torch.tensor(0.0)
        for name, task_output in outputs.items():
            if isinstance(task_output, tuple) and len(task_output) == 2:
                output, target = task_output
            else:
                output = task_output
                target = targets[model_output_dict[name].target]

            # TODO: How to handle task weights?
            # TODO: Should custom loss functions be passed to the model (like in Keras)
            #   Or to the model output (like in T4Rec)?
            loss = loss + model_output_dict[name].default_loss(output, target)

        return loss / len(model_output_dict)

    def initialize(self, data: Loader):
        if isinstance(data, Loader):
            self.double()  # TODO: Put in data-loader PR to standardize on float-32
            batch = sample_batch(data, batch_size=1, shuffle=False, include_targets=False)
        else:
            batch = data

        if isinstance(batch, torch.Tensor):
            device = batch.device
        elif isinstance(batch, tuple):
            device = batch[0].device
        elif isinstance(batch, dict):
            for d in batch.values():
                if isinstance(d, torch.Tensor):
                    device = d.device
                    break
        else:
            raise ValueError(f"Unsupported data type {type(batch)}")

        self.to(device)
        return self(batch)

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters())

    def test(self, mode: bool = True):
        self.train(False)
        self.testing = mode

        for child in module_utils.get_all_children(self):
            child.register_buffer("testing", torch.tensor(mode), persistent=False)

        return self

    @property
    def model_outputs(self) -> List[ModelOutput]:
        return module_utils.find_all_instances(self, ModelOutput)

    @property
    def model_outputs_by_name(self) -> Dict[str, ModelOutput]:
        return {out.name: out for out in self.model_outputs}

    @property
    def first(self) -> nn.Module:
        return self.blocks[0]

    @property
    def last(self) -> nn.Module:
        return self.blocks[-1]

    @property
    def input_schema(self) -> Schema:
        if self.schema:
            return self.schema

        input_schemas = []
        for child in module_utils.get_all_children(self):
            if hasattr(child, "input_schema"):
                input_schemas.append(child.input_schema)

        if not input_schemas:
            raise ValueError("No input schema found")

        return reduce(lambda a, b: a + b, input_schemas)  # type: ignore

    @property
    def output_schema(self) -> Schema:
        output_schemas = []
        for child in module_utils.get_all_children(self):
            if hasattr(child, "output_schema"):
                output_schemas.append(child.output_schema)

        if not output_schemas:
            raise ValueError("No output schema found")

        return reduce(lambda a, b: a + b, output_schemas)  # type: ignore
