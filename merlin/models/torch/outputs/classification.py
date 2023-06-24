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
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torchmetrics import AUROC, Accuracy, AveragePrecision, Metric, Precision, Recall

import merlin.dtypes as md
from merlin.models.torch.inputs.embedding import EmbeddingTable
from merlin.models.torch.outputs.base import ModelOutput
from merlin.schema import ColumnSchema, Schema


def default_binary_metrics() -> Tuple[Metric]:
    """Returns the default metrics used for binary classification.

    Returns
    -------
    Tuple[Metric]
        A tuple containing the Metric objects for Accuracy, AUROC, Precision,
        and Recall, all configured for a binary classification task.
    """
    return (
        Accuracy(task="binary"),
        AUROC(task="binary"),
        Precision(task="binary"),
        Recall(task="binary"),
    )


def default_categorical_prediction_metrics(num_classes, k=10):
    """Returns the default metrics used for multi-class classification.

    Parameters
    ----------
    num_classes: int
        The number of classes in the multi-class classification task.
    k: int, optional
        The number of top predictions to consider for the top-k metrics, by default 10.

    Returns
    -------
    Tuple[Metric]
        A tuple containing the Metric objects for Average Precision, Precision,
        and Recall, all configured for a multi-class classification task.
    """
    return (
        AveragePrecision(task="multiclass", num_classes=num_classes),
        Precision(task="multiclass", num_classes=num_classes),
        Recall(task="multiclass", num_classes=num_classes),
    )


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

    def __init__(
        self,
        schema: Optional[ColumnSchema] = None,
        loss: nn.Module = nn.BCEWithLogitsLoss(),
        metrics: Sequence[Metric] = default_binary_metrics(),
    ):
        """Initializes a BinaryOutput object."""
        super().__init__(
            nn.LazyLinear(1),
            nn.Sigmoid(),
            schema=schema,
            loss=loss,
            metrics=metrics,
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


class CategoricalTarget(nn.Module):
    """Prediction of a categorical feature.

    Parameters
    --------------
    col_schema: ColumnSchema, optional
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
        col_schema: Optional[ColumnSchema] = None,
        activation=None,
        bias: bool = True,
    ):
        super().__init__()

        if col_schema:
            self.schema = col_schema
            self.num_classes = col_schema.int_domain.max + 1
        else:
            self.num_classes = 2

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
        device = self.table.table.weight.device
        self.bias = nn.Parameter(torch.zeros(self.num_classes, dtype=torch.float32, device=device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model using input tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor for the forward pass.

        Returns
        ----------
        torch.Tensor
            Output tensor of the forward pass.
        """
        return torch.matmul(inputs, self.embeddings().t()).add(self.bias)

    def embeddings(self):
        """
        Fetch the weight matrix from the embedding table.

        Returns
        ----------
        nn.Parameter
            Weight matrix from the embedding table.
        """
        return self.table.table.weight

    def embedding_lookup(self, ids):
        """
        Fetch the embeddings for given indices from the embedding table.

        Parameters
        ----------
        ids : torch.Tensor
            Tensor containing indices for which embeddings are to be returned.

        Returns
        ----------
        torch.Tensor
            The corresponding embeddings.
        """
        return self.table.table(ids)


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
    """

    def __init__(
        self,
        to_call: Optional[
            Union[ColumnSchema, EmbeddingTable, CategoricalTarget, EmbeddingTablePrediction]
        ] = None,
        loss: nn.Module = nn.CrossEntropyLoss(),
        metrics: Optional[Sequence[Metric]] = None,
    ):
        if to_call is None or isinstance(to_call, ColumnSchema):
            _to_call = CategoricalTarget(to_call)
            schema = to_call
        elif isinstance(to_call, CategoricalTarget):
            _to_call = to_call
            schema = to_call.schema
        elif isinstance(to_call, (EmbeddingTable, EmbeddingTablePrediction)):
            if isinstance(to_call, EmbeddingTable):
                _to_call = EmbeddingTablePrediction(to_call)
            if len(_to_call.table.output_schema()) == 1:
                schema = _to_call.table.output_schema().first
            else:
                raise RuntimeError(
                    "The target column cannot be inferred because the output schema of "
                    f"{to_call.__class__.__name__} contains multiple columns."
                )
        else:
            raise ValueError(f"Unexpected type: {to_call.__class__.__name__}")

        self.num_classes = _to_call.num_classes

        if metrics is None:
            metrics = default_categorical_prediction_metrics(self.num_classes)

        super().__init__(
            _to_call,
            schema=schema,
            loss=loss,
            metrics=metrics,
        )

    def setup_schema(self, target: ColumnSchema):
        """Set up the schema for the output.

        Parameters
        ----------
        target: ColumnSchema
            The schema defining the column properties.
        """
        _target = target.with_dtype(md.float32)
        if "domain" not in target.properties:
            _target = _target.with_properties(
                {"domain": {"min": 0, "max": 1, "name": _target.name}},
            )
        if "value_count" not in target.properties:
            _target = _target.with_properties(
                {"value_count": {"min": self.num_classes, "max": self.num_classes}},
            )

        self.output_schema = Schema([_target])
