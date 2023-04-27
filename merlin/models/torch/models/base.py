from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn

from merlin.dataloader.torch import Loader
from merlin.io import Dataset

# from merlin.models.torch.base import register_post_hook, register_pre_hook
from merlin.models.torch.combinators import ParallelBlock
from merlin.models.torch.data import (  # needs_data_propagation_hook,; register_data_propagation_hook,
    initialize,
)
from merlin.models.torch.inputs.encoder import Encoder
from merlin.models.torch.outputs.base import ModelOutput
from merlin.models.torch.outputs.contrastive import ContrastiveOutput
from merlin.models.torch.predict import batch_predict
from merlin.models.torch.utils import module_utils
from merlin.schema import ColumnSchema, Schema, Tags


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
        self.testing = False

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

    def batch_predict(
        self, dataset: Dataset, batch_size: int, add_inputs: bool = True, index=None
    ) -> Dataset:
        return batch_predict(
            self, self.output_schema(), dataset, batch_size, add_inputs=add_inputs, index=index
        )

    def initialize(self, data: Loader):
        return initialize(self, data)

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

    # @property
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

    # @property
    def output_schema(self) -> Schema:
        output_schemas = []
        for child in module_utils.get_all_children(self):
            if hasattr(child, "output_schema"):
                output_schemas.append(child.output_schema)

        if not output_schemas:
            raise ValueError("No output schema found")

        return reduce(lambda a, b: a + b, output_schemas)  # type: ignore


class RetrievalModel(Model):
    DEFAULT_QUERY_NAME = "query"
    DEFAULT_CANDIDATE_NAME = "candidate"

    def __init__(
        self,
        *,
        query: Union[Encoder, nn.Module],
        output: Union[ModelOutput, nn.Module],
        candidate: Optional[Union[Encoder, nn.Module]] = None,
        pre=None,
        post=None,
    ):
        _query: Encoder = query if isinstance(query, Encoder) else Encoder(query)
        if query and candidate:
            _candidate: Encoder = (
                candidate if isinstance(candidate, Encoder) else Encoder(candidate)
            )
            query_name, candidate_name = self._query_branch_names(output)
            encoder = ParallelBlock({query_name: _query, candidate_name: _candidate})
        else:
            encoder = _query
            _candidate = Encoder(output)

        super().__init__(encoder, output, pre=pre, post=post)
        self.query = _query
        self.candidate = _candidate

    def _query_branch_names(self, output) -> Tuple[str, str]:
        query_name = self.DEFAULT_QUERY_NAME
        candidate_name = self.DEFAULT_CANDIDATE_NAME
        if isinstance(output, ContrastiveOutput):
            query_name = output.keys[0]
            candidate_name = output.keys[1]

        return query_name, candidate_name

    def query_embeddings(
        self,
        dataset: Optional[Dataset] = None,
        index: Optional[Union[str, ColumnSchema, Schema, Tags]] = None,
        batch_size=512,
        **kwargs,
    ) -> Dataset:
        if self.query.has_embedding_export and dataset is None:
            return self.query.export_embeddings(**kwargs)

        return self.query.encode(dataset, batch_size=batch_size, index=index, **kwargs)

    def candidate_embeddings(
        self,
        dataset: Optional[Dataset] = None,
        index: Optional[Union[str, ColumnSchema, Schema, Tags]] = None,
        batch_size=512,
        **kwargs,
    ) -> Dataset:
        if self.candidate.has_embedding_export and dataset is None:
            return self.candidate.export_embeddings(**kwargs)

        return self.candidate.encode(dataset, batch_size=batch_size, index=index, **kwargs)
