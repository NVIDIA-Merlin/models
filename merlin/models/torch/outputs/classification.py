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
import inspect
from typing import List, Optional, Sequence, Type, Union

import torch
import torchmetrics as tm
from torch import nn

import merlin.dtypes as md
from merlin.models.torch import schema
from merlin.models.torch.batch import Batch
from merlin.models.torch.inputs.embedding import EmbeddingTable
from merlin.models.torch.outputs.base import ModelOutput
from merlin.schema import ColumnSchema, Schema, Tags


class BinaryOutput(ModelOutput):
    """A prediction block for binary classification.

    Parameters
    ----------
    schema: Union[ColumnSchema, Schema], optional
        The schema defining the column properties. Default is None.
    loss: nn.Module
        The loss function used for training. Default is nn.BCEWithLogitsLoss().
    metrics: Sequence[Metric]
        The metrics used for evaluation. Default includes Accuracy, AUROC, Precision, and Recall.
    """

    DEFAULT_LOSS_CLS = nn.BCELoss
    DEFAULT_METRICS_CLS = (tm.Accuracy, tm.AUROC, tm.Precision, tm.Recall)

    def __init__(
        self,
        schema: Optional[ColumnSchema] = None,
        loss: Optional[nn.Module] = None,
        metrics: Sequence[tm.Metric] = (),
    ):
        """Initializes a BinaryOutput object."""
        super().__init__(
            nn.LazyLinear(1),
            nn.Sigmoid(),
            loss=loss or self.DEFAULT_LOSS_CLS(),
            metrics=metrics or [m(task="binary") for m in self.DEFAULT_METRICS_CLS],
        )
        if schema:
            self.initialize_from_schema(schema)
            self._initialized_from_schema = True

        if not self.metrics:
            self.metrics = self.default_metrics()

    def initialize_from_schema(self, target: Optional[Union[ColumnSchema, Schema]]):
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
    schema: Union[ColumnSchema, Schema], optional
        The schema defining the column properties. Default is None.
    loss : nn.Module, optional
        The loss function to use for the output model, defaults to
        torch.nn.CrossEntropyLoss.
    metrics : Optional[Sequence[Metric]], optional
        The metrics to evaluate the model output.
    logits_temperature: float, optional
        Parameter used to reduce model overconfidence, so that logits / T.
        by default 1.0
    """

    DEFAULT_LOSS_CLS = nn.CrossEntropyLoss
    DEFAULT_METRICS_CLS = (
        tm.RetrievalHitRate,
        tm.RetrievalNormalizedDCG,
        tm.RetrievalPrecision,
        tm.RetrievalRecall,
    )
    DEFAULT_K = (5,)

    def __init__(
        self,
        schema: Optional[Union[ColumnSchema, Schema]] = None,
        loss: Optional[nn.Module] = None,
        metrics: Optional[Sequence[tm.Metric]] = None,
        logits_temperature: float = 1.0,
    ):
        super().__init__(
            loss=loss or self.DEFAULT_LOSS_CLS(),
            metrics=metrics or create_retrieval_metrics(self.DEFAULT_METRICS_CLS, self.DEFAULT_K),
            logits_temperature=logits_temperature,
        )

        if schema:
            self.initialize_from_schema(schema)
            self._initialized_from_schema = True

    @classmethod
    def with_weight_tying(
        cls,
        block: nn.Module,
        selection: Optional[schema.Selection] = None,
        loss: nn.Module = nn.CrossEntropyLoss(),
        metrics: Optional[Sequence[tm.Metric]] = None,
        logits_temperature: float = 1.0,
    ) -> "CategoricalOutput":
        self = cls(loss=loss, metrics=metrics, logits_temperature=logits_temperature)
        self = self.tie_weights(block, selection)
        if not self.metrics:
            self.metrics = self.default_metrics(self.num_classes)

        return self

    def tie_weights(
        self, block: nn.Module, selection: Optional[schema.Selection] = None
    ) -> "CategoricalOutput":
        prediction = EmbeddingTablePrediction.with_weight_tying(block, selection)
        self.num_classes = prediction.num_classes
        if self:
            self[0] = prediction
        else:
            self.prepend(prediction)

        return self

    def initialize_from_schema(self, target: Optional[Union[ColumnSchema, Schema]]):
        """Set up the schema for the output.

        Parameters
        ----------
        target: Optional[ColumnSchema]
            The schema defining the column properties.
        """
        if not isinstance(target, (ColumnSchema, Schema)):
            raise ValueError(f"Target must be a ColumnSchema or Schema, got {target}.")

        if isinstance(target, Schema):
            if len(target) != 1:
                raise ValueError("Schema must contain exactly one column.")

            target = target.first

        to_call = CategoricalTarget(target)
        self.num_classes = to_call.num_classes
        self.prepend(to_call)

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

        self.target_name = col_schema.name
        self.num_classes = col_schema.int_domain.max + 1
        self.output_schema = categorical_output_schema(col_schema, self.num_classes)

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

    def should_apply_contrastive(self, batch: Optional[Batch]) -> bool:
        if batch is not None and batch.targets and self.training:
            return True

        return False


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

    def __init__(self, table: EmbeddingTable, selection: Optional[schema.Selection] = None):
        super().__init__()
        self.table = table
        if len(table.domains) > 1:
            if not selection:
                raise ValueError(
                    f"Table {table} has multiple columns. ",
                    "Must specify selection to choose column.",
                )
            self.add_selection(selection)
        else:
            self.num_classes = table.num_embeddings
            self.col_schema = table.input_schema.first
            self.col_name = self.col_schema.name
            self.target_name = self.col_name
        self.bias = nn.Parameter(
            torch.zeros(self.num_classes, dtype=torch.float32, device=self.embeddings().device)
        )
        self.output_schema = categorical_output_schema(self.col_schema, self.num_classes)

    @classmethod
    def with_weight_tying(
        cls,
        block: nn.Module,
        selection: Optional[schema.Selection] = None,
    ) -> "EmbeddingTablePrediction":
        if isinstance(block, EmbeddingTable):
            table = block
        else:
            if not selection:
                raise ValueError(
                    "Must specify a `selection` when providing a block that isn't a table."
                )

            try:
                selected = schema.select(block, selection)
                table = selected.leaf()
            except Exception as e:
                raise ValueError("Could not find embedding table in block.") from e

        return cls(table, selection)

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

    def add_selection(self, selection: schema.Selection):
        selected = schema.select(self.table.input_schema, selection)
        if not len(selected) == 1:
            raise ValueError("Schema must contain exactly one column. ", f"got: {selected}")
        self.col_schema = selected.first
        self.col_name = self.col_schema.name
        self.num_classes = self.col_schema.int_domain.max + 1
        self.output_schema = categorical_output_schema(self.col_schema, self.num_classes)
        self.target_name = self.col_name

        return self

    def embeddings(self) -> nn.Parameter:
        """Fetch the weight matrix from the embedding table.

        Returns
        ----------
        nn.Parameter
            Weight matrix from the embedding table.
        """
        if len(self.table.domains) > 1:
            return self.table.feature_weights(self.col_name)

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
        return self.table({self.col_name: inputs})[self.col_name]

    def should_apply_contrastive(self, batch: Optional[Batch]) -> bool:
        if batch is not None and batch.targets and self.training:
            return True

        return False


def categorical_output_schema(target: ColumnSchema, num_classes: int) -> Schema:
    """Return the output schema given the target column schema."""
    _target = target.with_dtype(md.float32)
    _target = _target.with_properties(
        {"domain": {"min": 0, "max": 1, "name": _target.name}},
    )
    if "value_count" not in target.properties:
        _target = _target.with_properties(
            {"value_count": {"min": num_classes, "max": num_classes}},
        )

    return Schema([_target])


def create_retrieval_metrics(
    metrics: Sequence[Type[tm.Metric]], ks: Sequence[int]
) -> List[tm.Metric]:
    """
    Create a list of retrieval metrics given metric types and a list of integers.
    For each integer in `ks`, a metric is created for each type in `metrics`.

    Parameters
    ----------
    metrics : Sequence[Type[tm.Metric]]
        The types of metrics to create. Each type should be a callable that
        accepts a single integer parameter `k` to instantiate a new metric.
    ks : Sequence[int]
        A list of integers to use as the `k` or `top_k` parameter when creating each metric.

    Returns
    -------
    List[tm.Metric]
        A list of metrics. The length of the list is equal to the product of
        the lengths of `metrics` and `ks`. The metrics are ordered first by
        the values in `ks`, then by the order in `metrics`.
    """

    outputs = []

    for k in ks:
        for metric in metrics:
            # check the parameters of the callable metric
            params = inspect.signature(metric).parameters

            # the argument name could be 'k' or 'top_k'
            arg_name = "top_k" if "top_k" in params else "k" if "k" in params else None

            if arg_name is not None:
                outputs.append(metric(**{arg_name: k}))
            else:
                raise ValueError(
                    "Expected a callable that accepts either ",
                    f"a 'k' or a 'top_k' parameter, but got {metric}",
                )

    return outputs
