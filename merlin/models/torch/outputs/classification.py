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
        if schema:
            self.setup_schema(schema)

        if not self.metrics:
            self.metrics = self.default_metrics()

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

    @classmethod
    def schema_selection(cls, schema: Schema) -> Schema:
        """Returns a schema containing all binary targets."""
        output = Schema()
        output += schema.select_by_tag([Tags.BINARY_CLASSIFICATION, Tags.BINARY])
        for col in schema.select_by_tag([Tags.CATEGORICAL]):
            if col.int_domain and col.int_domain.max == 1:
                output += Schema([col])

        return output


class CategoricalOutput(ModelOutput):
    """
    A prediction block for categorical targets.

    Parameters
    ----------
    to_call : Union[ColumnSchema, EmbeddingTable, CategoricalTarget,
              EmbeddingTablePrediction]
        The instance to be called for generating predictions.
    loss : nn.Module, optional
        The loss function to use for the output model, defaults to
        torch.nn.CrossEntropyLoss.
    metrics : Optional[Sequence[Metric]], optional
        The metrics to evaluate the model output.
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.0
    """

    def __init__(
        self,
        to_call: Optional[
            Union[
                Schema,
                ColumnSchema,
                EmbeddingTable,
                "CategoricalTarget",
                "EmbeddingTablePrediction",
            ]
        ] = None,
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
        elif isinstance(to_call, (EmbeddingTable)):
            self.prepend(EmbeddingTablePrediction(to_call))
        elif isinstance(to_call, (CategoricalTarget, EmbeddingTablePrediction)):
            self.prepend(to_call)
        else:
            raise ValueError(f"Invalid to_call type: {type(to_call)}")

        self.num_classes = self[0].num_classes

        if not self.metrics:
            self.metrics = self.default_metrics()

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
        if len(self) > 0 and isinstance(self[0], CategoricalTarget):
            self[0] = to_call
        else:
            self.prepend(to_call)
        self.output_schema = categorical_output_schema(target, self[0].num_classes)

    def default_metrics(self) -> List[Metric]:
        """Returns the default metrics used for multi-class classification."""
        return [
            AveragePrecision(task="multiclass", num_classes=self.num_classes),
            Precision(task="multiclass", num_classes=self.num_classes),
            Recall(task="multiclass", num_classes=self.num_classes),
        ]

    @classmethod
    def schema_selection(cls, schema: Schema) -> Schema:
        """Returns a schema containing all categorical targets."""
        output = Schema()
        for col in schema.select_by_tag([Tags.CATEGORICAL]):
            if col.int_domain and col.int_domain.max > 1:
                output += Schema([col])

        return output


class CategoricalTarget(nn.Module):
    """Prediction of a categorical feature.

    Parameters
    --------------
    feature: Union[ColumnSchema, Schema], optional
        Schema of the column being targeted. The schema must contain an
        'int_domain' specifying the maximum integer value representing the
        categorical classes.
    activation: callable, optional
        Activation function to be applied to the output of the linear layer.
        If None, no activation function is applied.
    bias: bool, default=True
        If set to False, the layer will not learn an additive bias.

    Returns
    ---------
    torch.Tensor
        The tensor output of the forward method.
    """

    def __init__(
        self,
        feature: Optional[Union[Schema, ColumnSchema]] = None,
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
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the module and applies the activation function if present.

        Parameters
        --------------
        inputs: torch.Tensor
            Input tensor for the forward pass.

        Returns
        ---------
        torch.Tensor
            Output tensor from the forward pass of the model.
        """
        output = self.linear(inputs)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def embedding_lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Selects the embeddings for the given indices.

        Parameters
        --------------
        ids: torch.Tensor
            Tensor containing indices for which embeddings are to be returned.

        Returns
        ---------
        torch.Tensor
            The corresponding embeddings.
        """
        return torch.index_select(self.embeddings(), 1, ids).t()

    def embeddings(self) -> nn.Parameter:
        """
        Returns the embeddings from the weight matrix.

        Returns
        ---------
        nn.Parameter
            The embeddings.
        """
        return self.linear.weight.t()


class EmbeddingTablePrediction(nn.Module):
    """Prediction of a categorical feature using weight-sharing [1] with an embedding table.

    Parameters
    ----------
    table : EmbeddingTable
        The embedding table to use as the weight matrix.

    References:
    ----------
    [1] Hakan Inan, Khashayar Khosravi, and Richard Socher. 2016. Tying word vectors
    and word classifiers: A loss framework for language modeling. arXiv preprint
    arXiv:1611.01462 (2016).
    """

    def __init__(self, table: EmbeddingTable):
        super().__init__()
        self.table = table
        self.num_classes = table.num_embeddings
        self.bias = nn.Parameter(
            torch.zeros(self.num_classes, dtype=torch.float32, device=self.embeddings().device)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model using input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor for the forward pass.

        Returns
        ----------
        torch.Tensor
            Output tensor of the forward pass.
        """
        return nn.functional.linear(inputs, self.embeddings(), self.bias)

    def embeddings(self) -> nn.Parameter:
        """Fetch the weight matrix from the embedding table.

        Returns
        ----------
        nn.Parameter
            Weight matrix from the embedding table.
        """
        return self.table.table.weight

    def embedding_lookup(self, inputs: torch.Tensor) -> torch.Tensor:
        """Fetch the embeddings for given indices from the embedding table.

        Parameters
        ----------
        ids : torch.Tensor
            Tensor containing indices for which embeddings are to be returned.

        Returns
        ----------
        torch.Tensor
            The corresponding embeddings.
        """
        # TODO: Make sure that we check if the table holds multiple features
        # If so, we need to add domain.min to the inputs
        return self.table.table(inputs)


def categorical_output_schema(target: ColumnSchema, num_classes: int) -> Schema:
    """Return the output schema given the target column schema."""
    _target = target.with_dtype(md.float32)
    if "domain" not in target.properties:
        _target = _target.with_properties(
            {"domain": {"min": 0, "max": 1, "name": _target.name}},
        )
    if "value_count" not in target.properties:
        _target = _target.with_properties(
            {"value_count": {"min": num_classes, "max": num_classes}},
        )

    return Schema([_target])
