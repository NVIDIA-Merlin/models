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

import torch
from torch import nn
from torchmetrics import AUROC, Accuracy, Metric, Precision, Recall

import merlin.dtypes as md
from merlin.core.dispatch import DataFrameType
from merlin.io import Dataset
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


class CategoricalOutput(ModelOutput):
    def __init__(
        self,
        to_call: Union[
            Schema, ColumnSchema, EmbeddingTable, "CategoricalTarget", "EmbeddingTablePrediction"
        ],
        loss=nn.CrossEntropyLoss(),
        metrics: Sequence[Metric] = (),
        # logits_temperature: float = 1.0,
    ):
        # if to_call is not None:
        #     if isinstance(to_call, (Schema, ColumnSchema)):
        #         _to_call = CategoricalTarget(to_call)
        #         if isinstance(to_call, Schema):
        #             to_call = to_call.first
        #         target_name = target_name or to_call.name
        #     elif isinstance(to_call, EmbeddingTable):
        #         _to_call = EmbeddingTablePrediction(to_call)
        #         if len(to_call.schema) == 1:
        #             target_name = _to_call.table.schema.first.name
        #         else:
        #             raise ValueError("Can't infer the target automatically, please provide it.")
        #     else:
        #         _to_call = to_call

        super().__init__(
            to_call if to_call else (),
            loss=loss,
            metrics=metrics,
            # logits_temperature=logits_temperature,
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

        _target = target.with_dtype(md.float32).with_tags([Tags.CONTINUOUS])

        self.output_schema = Schema([_target])

    def create_output_schema(self, target: ColumnSchema) -> Schema:
        return categorical_output_schema(target, self.to_call.num_classes)

    def to_dataset(self, gpu=None) -> Dataset:
        return self.to_call.to_dataset(gpu=gpu)

    def to_df(self, gpu=None) -> DataFrameType:
        return self.to_call.to_df(gpu=gpu)


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

    # def to_dataset(self, gpu=None) -> Dataset:
    #     return Dataset(self.to_df(gpu=gpu))

    # def to_df(self, gpu=None) -> DataFrameType:
    #     return tensor_to_df(self.linear.weight, gpu=gpu)

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
            torch.empty(self.num_classes, dtype=torch.float32, device=table.weight.device)
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

    # def to_dataset(self, gpu=None) -> Dataset:
    #     return self.table.to_dataset(gpu=gpu)

    # def to_df(self, gpu=None) -> DataFrameType:
    #     return self.table.to_df(gpu=gpu)


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
