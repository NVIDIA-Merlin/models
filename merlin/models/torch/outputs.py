from typing import Dict, Optional, Sequence, Union, List

import torch
from torch import nn
from torchmetrics import Accuracy, AUROC, Metric, Precision, Recall

import merlin.dtypes as md
from merlin.schema import ColumnSchema, Schema
from merlin.models.torch.block import Block


class ModelOutput(Block):
    def __init__(
        self, 
        *module: nn.Module,
        loss: nn.Module,
        metrics: Sequence[Metric] = (),
        name: Optional[str] = None,
        schema: Optional[ColumnSchema] = None,
    ):
        super().__init__(*module, name=name)
        self.loss = loss
        self.metrics = metrics
        if schema:
            self.setup_schema(schema)
        self.register_buffer("target", torch.zeros(1, dtype=torch.float32))
        
    def setup_schema(self, schema: ColumnSchema):
        self.schema = schema
        
    def eval(self):
        # Reset target
        self.target = torch.zeros(1, dtype=torch.float32)

        return self.train(False)


class BinaryOutput(ModelOutput):
    def __init__(
        self,
        loss=nn.BCEWithLogitsLoss(),
        metrics: Sequence[Metric] = (
            Accuracy(task="binary"),
            AUROC(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
        ),
        schema: Optional[ColumnSchema] = None,
    ):
        super().__init__(
            nn.LazyLinear(1), 
            nn.Sigmoid(),
            loss=loss,
            metrics=metrics,
            schema=schema,
        )
        
    def setup_schema(self, target: ColumnSchema):
        _target = target.with_dtype(md.float32)
        if "domain" not in target.properties:
            _target = _target.with_properties(
                {"domain": {"min": 0, "max": 1, "name": _target.name}},
            )

        self.output_schema = Schema([_target])


def compute_loss(
    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
    targets: Union[torch.Tensor, Dict[str, torch.Tensor]], 
    model_outputs: List[ModelOutput],
    compute_metrics: bool = False
):
    """
    Update targets with model_out.target for each model_out
    """
    
    raise NotImplementedError()
