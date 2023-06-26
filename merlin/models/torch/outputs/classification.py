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
from typing import List, Optional, Sequence, Union

import torch
from torch import nn
from torchmetrics import AUROC, Accuracy, AveragePrecision, Metric, Precision, Recall

import merlin.dtypes as md
from merlin.models.torch.inputs.embedding import EmbeddingTable
from merlin.models.torch.outputs.base import ModelOutput
from merlin.schema import ColumnSchema, Schema, Tags


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
        if not self.metrics:
            self.metrics = self.default_metrics()

    @classmethod
    def schema_selection(cls, schema: Schema) -> Schema:
        """Returns a schema containing all binary targets."""
        output = Schema()
        output += schema.select_by_tag([Tags.BINARY_CLASSIFICATION, Tags.BINARY])
        for col in schema.select_by_tag([Tags.CATEGORICAL]):
            if col.int_domain and col.int_domain.max == 1:
                output += col

        return output


class CategoricalOutput(ModelOutput):
    def __init__(
        self,
        to_call: Union[
            Schema, ColumnSchema, EmbeddingTable, "CategoricalTarget", "EmbeddingTablePrediction"
        ],
        loss=nn.CrossEntropyLoss(),
        metrics: Optional[Sequence[Metric]] = None,
        logits_temperature: float = 1.0,
    ):
        super().__init__(
            loss=loss,
            metrics=metrics,
            logits_temperature=logits_temperature,
        )

        if isinstance(to_call, (Schema, ColumnSchema)):
            self.setup_schema(to_call)
        elif isinstance(to_call, EmbeddingTable):
            self.prepend(EmbeddingTablePrediction(to_call))
        elif isinstance(to_call, (CategoricalTarget, EmbeddingTablePrediction)):
            self.prepend(to_call)
        else:
            raise ValueError(f"Invalid to_call type: {type(to_call)}")

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
        to_call = CategoricalTarget(target)
        if isinstance(self[0], CategoricalTarget):
            self[0] = to_call
        else:
            self.prepend(to_call)
        self.output_schema = categorical_output_schema(target, self[0].num_classes)
        self.num_classes = to_call.num_classes

        if not self.metrics:
            self.metrics = self.default_metrics()

    def default_metrics(self) -> List[Metric]:
        return (
            AveragePrecision(task="multiclass", num_classes=self.num_classes),
            Precision(task="multiclass", num_classes=self.num_classes),
            Recall(task="multiclass", num_classes=self.num_classes),
        )

    @classmethod
    def schema_selection(cls, schema: Schema) -> Schema:
        """Returns a schema containing all categorical targets."""
        output = Schema()
        for col in schema.select_by_tag([Tags.CATEGORICAL]):
            if col.int_domain and col.int_domain.max > 1:
                output += col

        return output


class CategoricalTarget(nn.Module):
    def __init__(
        self,
        feature: Union[Schema, ColumnSchema] = None,
        activation=None,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(feature, Schema):
            assert len(feature) == 1, "Schema can have max 1 feature"
            col_schema = feature.first
        else:
            col_schema = feature

        self.schema = col_schema
        self.target_name = col_schema.name
        self.num_classes = col_schema.int_domain.max + 1

        self.linear = nn.LazyLinear(self.num_classes, bias=bias)
        if activation is not None:
            self.activation = activation()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.linear(inputs)
        if hasattr(self, "activation"):
            output = self.activation(output)

        return output

    def embedding_lookup(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.index_select(self.embeddings(), 1, ids).t()

    def embeddings(self) -> nn.Parameter:
        return self.linear.weight.t()

    @property
    def is_initialized(self) -> bool:
        return not isinstance(self.linear, nn.LazyLinear)


class EmbeddingTablePrediction(nn.Module):
    """Prediction of a categorical feature using weight-sharing [1] with an embedding table

    Parameters
    ----------
    table : EmbeddingTable
        The embedding table to use as the weight matrix
    bias_initializer : str, optional
        Initializer for the bias vector, by default "zeros"

    References:
    ----------
    [1] Hakan Inan, Khashayar Khosravi, and Richard Socher. 2016. Tying word vectors
    and word classifiers: A loss framework for language modeling. arXiv preprint
    arXiv:1611.01462 (2016).
    """

    def __init__(self, table: EmbeddingTable, bias_initializer="zeros"):
        super().__init__()
        self.table = table
        self.num_classes = table.num_embeddings
        self.bias_initializer = bias_initializer
        self.bias = nn.Parameter(
            torch.empty(self.num_classes, dtype=torch.float32, device=self.embeddings().device)
        )

    def reset_parameters(self) -> None:
        if self.bias_initializer == "zeros":
            nn.init.constant_(self.bias, 0)
        else:
            raise ValueError(f"Unknown initializer {self.bias_initializer}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(inputs, self.embeddings(), self.bias)

    def embeddings(self) -> nn.Parameter:
        return self.table.table.weight

    def embedding_lookup(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO: Make sure that we check if the table holds multiple features
        # If so, we need to add domain.min to the inputs
        return self.table.table(inputs)


def _fix_shape_and_dtype(output, target):
    if len(output.shape) == len(target.shape) + 1 and output.shape[-1] == 1:
        output = output.squeeze(-1)

    return output, target.type_as(output)


def categorical_output_schema(target: ColumnSchema, num_classes: int) -> Schema:
    """Return the output schema given the target column schema."""
    _target = target.with_dtype(md.float32)
    if "domain" not in _target.properties:
        _target = _target.with_properties(
            {
                "domain": {"min": 0, "max": 1.0, "name": _target.name},
                "value_count": {"min": num_classes, "max": num_classes},
            }
        )

    return Schema([_target])
